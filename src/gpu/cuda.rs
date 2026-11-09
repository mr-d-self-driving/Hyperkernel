use std::ffi::{CString, c_void};
use std::os::fd::{AsRawFd, BorrowedFd, FromRawFd, OwnedFd};
use std::ptr;
use std::sync::OnceLock;

use cudarc::driver::sys::{self, CUlaunchAttributeID, CUresult};

pub use sys::CUfunction_attribute as Attribute;
pub use sys::CUlimit as Limit;
pub use sys::CUtensorMap as Map;
pub use sys::CUtensorMapDataType as MapElem;
pub use sys::CUtensorMapSwizzle as MapSwizzle;
pub use sys::{CUdevice as Device, CUmemGenericAllocationHandle as Handle};

#[derive(Clone, Copy, Debug)]
pub struct Stream(sys::CUstream);

#[derive(Clone, Copy, Debug)]
pub struct Module(sys::CUmodule);

#[derive(Clone, Copy, Debug)]
pub struct Function(sys::CUfunction);

static CONTEXT: OnceLock<(usize, Device)> = OnceLock::new();

fn posix() -> sys::CUmemAllocationHandleType {
    sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
}

fn ok(r: CUresult, call: &str) -> Result<(), String> {
    r.result().map_err(|e| format!("{call}: {e}"))
}

fn driver() -> Result<(), String> {
    match unsafe { sys::is_culib_present() } {
        true => ok(unsafe { sys::cuInit(0) }, "cuInit"),
        false => Err("no libcuda.so on the library path: install the NVIDIA driver (R580 or newer)".into()),
    }
}

pub fn visible(need: usize) -> Result<(), String> {
    driver()?;
    let mut n = 0;
    ok(unsafe { sys::cuDeviceGetCount(&mut n) }, "cuDeviceGetCount")?;
    let set =
        std::env::var("CUDA_VISIBLE_DEVICES").map_or_else(|_| "unset".to_string(), |v| format!("{v:?}"));
    match n as usize >= need {
        true => Ok(()),
        false => Err(format!("{n} GPUs visible (CUDA_VISIBLE_DEVICES {set}), this model runs on {need}")),
    }
}

pub fn init(ordinal: usize) -> Result<Device, String> {
    if let Some(&(_, d)) = CONTEXT.get() {
        bind()?;
        return Ok(d);
    }
    driver()?;
    let (mut device, mut ctx) = (0, ptr::null_mut());
    unsafe {
        ok(sys::cuDeviceGet(&mut device, ordinal as i32), "cuDeviceGet")?;
        ok(sys::cuDevicePrimaryCtxRetain(&mut ctx, device), "cuDevicePrimaryCtxRetain")?;
        ok(sys::cuCtxSetCurrent(ctx), "cuCtxSetCurrent")?;
    }
    let _ = CONTEXT.set((ctx as usize, device));
    Ok(device)
}

pub fn bind() -> Result<(), String> {
    let &(ctx, _) = CONTEXT.get().ok_or("cuCtxSetCurrent: no context (cuda::init first)")?;
    ok(unsafe { sys::cuCtxSetCurrent(ctx as sys::CUcontext) }, "cuCtxSetCurrent")
}

pub fn device() -> Result<Device, String> {
    CONTEXT.get().map(|c| c.1).ok_or_else(|| "no context (cuda::init first)".to_string())
}

pub fn attribute(a: sys::CUdevice_attribute) -> Result<i32, String> {
    let mut v = 0;
    ok(unsafe { sys::cuDeviceGetAttribute(&mut v, a, device()?) }, "cuDeviceGetAttribute")?;
    Ok(v)
}

pub fn stream() -> Result<Stream, String> {
    let mut s = ptr::null_mut();
    ok(
        unsafe { sys::cuStreamCreate(&mut s, sys::CUstream_flags::CU_STREAM_NON_BLOCKING as u32) },
        "cuStreamCreate",
    )?;
    Ok(Stream(s))
}

pub fn after(waits: Stream, on: Stream, e: &Event) -> Result<(), String> {
    ok(unsafe { sys::cuEventRecord(e.0, on.0) }, "cuEventRecord")?;
    ok(unsafe { sys::cuStreamWaitEvent(waits.0, e.0, 0) }, "cuStreamWaitEvent")
}

pub fn sync(s: Stream) -> Result<(), String> {
    ok(unsafe { sys::cuStreamSynchronize(s.0) }, "cuStreamSynchronize")
}

pub fn sync_all() -> Result<(), String> {
    ok(unsafe { sys::cuCtxSynchronize() }, "cuCtxSynchronize")
}

pub fn free_memory() -> Result<u64, String> {
    let (mut free, mut total) = (0, 0);
    ok(unsafe { sys::cuMemGetInfo_v2(&mut free, &mut total) }, "cuMemGetInfo")?;
    Ok(free as u64)
}

pub fn host_alloc(len: usize) -> Result<(*mut u8, u64), String> {
    bind()?;
    let (mut p, mut d) = (ptr::null_mut(), 0);
    let flags = sys::CU_MEMHOSTALLOC_DEVICEMAP | sys::CU_MEMHOSTALLOC_PORTABLE;
    unsafe {
        ok(sys::cuMemHostAlloc(&mut p, len.max(1), flags), "cuMemHostAlloc")?;
        ptr::write_bytes(p as *mut u8, 0, len.max(1));
        ok(sys::cuMemHostGetDevicePointer_v2(&mut d, p, 0), "cuMemHostGetDevicePointer")?;
    }
    Ok((p as *mut u8, d))
}

pub unsafe fn host_free(p: *mut u8) {
    let _ = unsafe { sys::cuMemFreeHost(p as *mut c_void) };
}

pub unsafe fn host_register(p: *mut u8, len: usize) -> Result<u64, String> {
    let mut d = 0;
    let flags = sys::CU_MEMHOSTREGISTER_DEVICEMAP | sys::CU_MEMHOSTREGISTER_PORTABLE;
    unsafe {
        ok(sys::cuMemHostRegister_v2(p as *mut c_void, len, flags), "cuMemHostRegister")?;
        ok(sys::cuMemHostGetDevicePointer_v2(&mut d, p as *mut c_void, 0), "cuMemHostGetDevicePointer")?;
    }
    Ok(d)
}

pub fn alloc(len: usize) -> Result<u64, String> {
    let mut d = 0;
    ok(unsafe { sys::cuMemAlloc_v2(&mut d, len.max(1)) }, "cuMemAlloc")?;
    memset8(d, 0, len.max(1))?;
    Ok(d)
}

pub fn memset8(d: u64, byte: u8, len: usize) -> Result<(), String> {
    ok(unsafe { sys::cuMemsetD8_v2(d, byte, len) }, "cuMemsetD8")
}

pub fn memset32(d: u64, word: u32, n: usize) -> Result<(), String> {
    ok(unsafe { sys::cuMemsetD32_v2(d, word, n) }, "cuMemsetD32")
}

pub fn htod(d: u64, bytes: &[u8]) -> Result<(), String> {
    if bytes.is_empty() {
        return Ok(());
    }
    ok(unsafe { sys::cuMemcpyHtoD_v2(d, bytes.as_ptr() as *const c_void, bytes.len()) }, "cuMemcpyHtoD")
}

pub fn dtoh(out: &mut [u8], d: u64) -> Result<(), String> {
    if out.is_empty() {
        return Ok(());
    }
    ok(unsafe { sys::cuMemcpyDtoH_v2(out.as_mut_ptr() as *mut c_void, d, out.len()) }, "cuMemcpyDtoH")
}

pub struct Event(sys::CUevent);

pub fn event(s: Stream) -> Result<Event, String> {
    let mut e = ptr::null_mut();
    unsafe {
        ok(sys::cuEventCreate(&mut e, 0), "cuEventCreate")?;
        ok(sys::cuEventRecord(e, s.0), "cuEventRecord")?;
    }
    Ok(Event(e))
}

impl Event {
    pub fn reached(&self) -> Result<bool, String> {
        match unsafe { sys::cuEventQuery(self.0) } {
            CUresult::CUDA_ERROR_NOT_READY => Ok(false),
            r => ok(r, "cuEventQuery").map(|()| true),
        }
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        let _ = unsafe { sys::cuEventDestroy_v2(self.0) };
    }
}

pub fn dtod(to: u64, from: u64, len: usize) -> Result<(), String> {
    ok(unsafe { sys::cuMemcpyDtoD_v2(to, from, len) }, "cuMemcpyDtoD")
}

pub fn dtod_async(to: u64, from: u64, len: usize, s: Stream) -> Result<(), String> {
    ok(unsafe { sys::cuMemcpyDtoDAsync_v2(to, from, len, s.0) }, "cuMemcpyDtoDAsync")
}

pub fn memset32_async(d: u64, word: u32, n: usize, s: Stream) -> Result<(), String> {
    ok(unsafe { sys::cuMemsetD32Async(d, word, n, s.0) }, "cuMemsetD32Async")
}

pub fn module(image: &[u8]) -> Result<Module, String> {
    let mut m = ptr::null_mut();
    ok(unsafe { sys::cuModuleLoadData(&mut m, image.as_ptr() as *const c_void) }, "cuModuleLoadData")?;
    Ok(Module(m))
}

pub fn function(m: Module, name: &str) -> Result<Function, String> {
    let c = CString::new(name).map_err(|_| format!("a function name with a NUL: {name}"))?;
    let mut f = ptr::null_mut();
    ok(unsafe { sys::cuModuleGetFunction(&mut f, m.0, c.as_ptr()) }, &format!("cuModuleGetFunction {name}"))?;
    ok(unsafe { sys::cuFuncLoad(f) }, &format!("cuFuncLoad {name}"))?;
    Ok(Function(f))
}

pub fn global(m: Module, name: &str) -> Result<Option<(u64, usize)>, String> {
    let c = CString::new(name).map_err(|_| format!("a global name with a NUL: {name}"))?;
    let (mut d, mut n) = (0, 0);
    match unsafe { sys::cuModuleGetGlobal_v2(&mut d, &mut n, m.0, c.as_ptr()) } {
        CUresult::CUDA_ERROR_NOT_FOUND => Ok(None),
        r => ok(r, &format!("cuModuleGetGlobal {name}")).map(|()| Some((d, n))),
    }
}

pub fn get_attribute(f: Function, a: Attribute) -> Result<i32, String> {
    let mut v = 0;
    ok(unsafe { sys::cuFuncGetAttribute(&mut v, a, f.0) }, &format!("cuFuncGetAttribute {a:?}"))?;
    Ok(v)
}

pub fn set_attribute(f: Function, a: Attribute, v: i32) -> Result<(), String> {
    ok(unsafe { sys::cuFuncSetAttribute(f.0, a, v) }, &format!("cuFuncSetAttribute {a:?}"))
}

pub fn limit(l: Limit) -> Result<usize, String> {
    let mut v = 0;
    ok(unsafe { sys::cuCtxGetLimit(&mut v, l) }, &format!("cuCtxGetLimit {l:?}"))?;
    Ok(v)
}

pub fn set_limit(l: Limit, v: usize) -> Result<(), String> {
    ok(unsafe { sys::cuCtxSetLimit(l, v) }, &format!("cuCtxSetLimit {l:?}"))
}

pub fn tensor_map(
    elem: u32,
    addr: u64,
    dims: &[u64],
    strides: &[u64],
    tile: &[u32],
    swizzle: u32,
    corners: Option<(&[i32], &[i32])>,
) -> Result<[u8; size_of::<Map>()], String> {
    let boxed = match corners {
        None => tile.len() == dims.len(),
        Some((lower, upper)) => {
            tile.len() == 2
                && dims.len() >= 3
                && lower.len() + 2 == dims.len()
                && upper.len() + 2 == dims.len()
        }
    };
    if elem > 15
        || swizzle > 6
        || dims.is_empty()
        || dims.len() > 5
        || strides.len() + 1 != dims.len()
        || !boxed
    {
        return Err(format!(
            "cuTensorMapEncode: elem {elem}, swizzle {swizzle}, {} dims, {} strides, {} box, corners {corners:?}",
            dims.len(),
            strides.len(),
            tile.len()
        ));
    }
    let mut map = Map { opaque: [0; 16] };
    let ones = [1u32; 5];
    let (elem, swizzle): (sys::CUtensorMapDataType, sys::CUtensorMapSwizzle) = unsafe {
        (
            std::mem::transmute::<u32, sys::CUtensorMapDataType>(elem),
            std::mem::transmute::<u32, sys::CUtensorMapSwizzle>(swizzle),
        )
    };
    let (interleave, promote, fill) = (
        sys::CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        sys::CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        sys::CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    );
    let (rank, at) = (dims.len() as u32, addr as *mut c_void);
    match corners {
        None => ok(
            unsafe {
                sys::cuTensorMapEncodeTiled(
                    &mut map,
                    elem,
                    rank,
                    at,
                    dims.as_ptr(),
                    strides.as_ptr(),
                    tile.as_ptr(),
                    ones.as_ptr(),
                    interleave,
                    swizzle,
                    promote,
                    fill,
                )
            },
            "cuTensorMapEncodeTiled",
        )?,
        Some((lower, upper)) => ok(
            unsafe {
                sys::cuTensorMapEncodeIm2col(
                    &mut map,
                    elem,
                    rank,
                    at,
                    dims.as_ptr(),
                    strides.as_ptr(),
                    lower.as_ptr(),
                    upper.as_ptr(),
                    tile[0],
                    tile[1],
                    ones.as_ptr(),
                    interleave,
                    swizzle,
                    promote,
                    fill,
                )
            },
            "cuTensorMapEncodeIm2col",
        )?,
    }
    Ok(unsafe { std::mem::transmute::<Map, [u8; size_of::<Map>()]>(map) })
}

#[derive(Clone, Copy, Debug)]
pub enum Launch {
    Cooperative,
    Cluster(u32),
    Window(u64, usize),
    Dependent,
}

pub fn launch(
    f: Function,
    grid: u32,
    block: u32,
    smem: u32,
    s: Stream,
    params: &[u64],
    with: &[Launch],
) -> Result<(), String> {
    let mut attrs: Vec<sys::CUlaunchAttribute> = with
        .iter()
        .map(|w| {
            let mut a: sys::CUlaunchAttribute = unsafe { std::mem::zeroed() };
            match *w {
                Launch::Cooperative => {
                    a.id = CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_COOPERATIVE;
                    a.value.cooperative = 1;
                }
                Launch::Cluster(x) => {
                    a.id = CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
                    a.value.clusterDim = sys::CUlaunchAttributeValue_union__bindgen_ty_1 { x, y: 1, z: 1 };
                }
                Launch::Window(at, len) => {
                    a.id = CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW;
                    a.value.accessPolicyWindow = sys::CUaccessPolicyWindow {
                        base_ptr: at as *mut c_void,
                        num_bytes: len,
                        hitRatio: 1.0,
                        hitProp: sys::CUaccessProperty::CU_ACCESS_PROPERTY_PERSISTING,
                        missProp: sys::CUaccessProperty::CU_ACCESS_PROPERTY_STREAMING,
                    };
                }
                Launch::Dependent => {
                    a.id = CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
                    a.value.programmaticStreamSerializationAllowed = 1;
                }
            }
            a
        })
        .collect();
    let mut words = params.to_vec();
    let mut ptrs: Vec<*mut c_void> = words.iter_mut().map(|w| w as *mut u64 as *mut c_void).collect();
    let config = sys::CUlaunchConfig {
        gridDimX: grid,
        gridDimY: 1,
        gridDimZ: 1,
        blockDimX: block,
        blockDimY: 1,
        blockDimZ: 1,
        sharedMemBytes: smem,
        hStream: s.0,
        attrs: attrs.as_mut_ptr(),
        numAttrs: attrs.len() as u32,
    };
    ok(unsafe { sys::cuLaunchKernelEx(&config, f.0, ptrs.as_mut_ptr(), ptr::null_mut()) }, "cuLaunchKernelEx")
}

pub fn launch_packed(f: Function, grid: u32, block: u32, s: Stream, args: &[u64]) -> Result<(), String> {
    let mut buf = args.to_vec();
    let mut size = buf.len() * 8;
    let mut extra = [
        sys::CU_LAUNCH_PARAM_BUFFER_POINTER_AS_INT as usize as *mut c_void,
        buf.as_mut_ptr() as *mut c_void,
        sys::CU_LAUNCH_PARAM_BUFFER_SIZE_AS_INT as usize as *mut c_void,
        &mut size as *mut usize as *mut c_void,
        sys::CU_LAUNCH_PARAM_END_AS_INT as usize as *mut c_void,
    ];
    ok(
        unsafe {
            sys::cuLaunchKernel(f.0, grid, 1, 1, block, 1, 1, 0, s.0, ptr::null_mut(), extra.as_mut_ptr())
        },
        "cuLaunchKernel",
    )
}

fn prop(export: bool) -> Result<sys::CUmemAllocationProp, String> {
    let mut p: sys::CUmemAllocationProp = unsafe { std::mem::zeroed() };
    p.type_ = sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
    p.location =
        sys::CUmemLocation { type_: sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE, id: device()? };
    if export {
        p.requestedHandleTypes = posix();
        p.allocFlags.gpuDirectRDMACapable = u8::from(
            attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED)?
                != 0,
        );
    }
    Ok(p)
}

pub fn granularity(export: bool) -> Result<usize, String> {
    let mut g = 0;
    let p = prop(export)?;
    ok(
        unsafe {
            sys::cuMemGetAllocationGranularity(
                &mut g,
                &p,
                sys::CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
            )
        },
        "cuMemGetAllocationGranularity",
    )?;
    Ok(g)
}

pub fn reserve(len: usize, align: usize) -> Result<u64, String> {
    let mut va = 0;
    ok(unsafe { sys::cuMemAddressReserve(&mut va, len, align, 0, 0) }, "cuMemAddressReserve")?;
    Ok(va)
}

pub fn create(len: usize, export: bool) -> Result<Handle, String> {
    let mut h = 0;
    let p = prop(export)?;
    ok(unsafe { sys::cuMemCreate(&mut h, len, &p, 0) }, "cuMemCreate")?;
    Ok(h)
}

pub fn map(va: u64, len: usize, h: Handle) -> Result<(), String> {
    ok(unsafe { sys::cuMemMap(va, len, 0, h, 0) }, "cuMemMap")?;
    let access = sys::CUmemAccessDesc {
        location: sys::CUmemLocation {
            type_: sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
            id: device()?,
        },
        flags: sys::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
    };
    ok(unsafe { sys::cuMemSetAccess(va, len, &access, 1) }, "cuMemSetAccess")
}

pub fn export(h: Handle) -> Result<OwnedFd, String> {
    let mut fd: i32 = -1;
    ok(
        unsafe { sys::cuMemExportToShareableHandle(&mut fd as *mut i32 as *mut c_void, h, posix(), 0) },
        "cuMemExportToShareableHandle",
    )?;
    Ok(unsafe { OwnedFd::from_raw_fd(fd) })
}

pub fn import(fd: BorrowedFd<'_>) -> Result<Handle, String> {
    let mut h = 0;
    ok(
        unsafe {
            sys::cuMemImportFromShareableHandle(&mut h, fd.as_raw_fd() as usize as *mut c_void, posix())
        },
        "cuMemImportFromShareableHandle",
    )?;
    Ok(h)
}

fn multicast_prop(devices: usize, len: usize) -> sys::CUmulticastObjectProp {
    sys::CUmulticastObjectProp {
        numDevices: devices as u32,
        size: len,
        handleTypes: posix().0 as u64,
        flags: 0,
    }
}

pub fn multicast_granularity(devices: usize) -> Result<usize, String> {
    if attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED)? == 0 {
        return Ok(0);
    }
    let mut g = 0;
    let p = multicast_prop(devices, 0);
    ok(
        unsafe {
            sys::cuMulticastGetGranularity(
                &mut g,
                &p,
                sys::CUmulticastGranularity_flags::CU_MULTICAST_GRANULARITY_MINIMUM,
            )
        },
        "cuMulticastGetGranularity",
    )?;
    Ok(g)
}

pub fn multicast_create(devices: usize, len: usize) -> Result<Handle, String> {
    let mut h = 0;
    let p = multicast_prop(devices, len);
    ok(unsafe { sys::cuMulticastCreate(&mut h, &p) }, "cuMulticastCreate")?;
    Ok(h)
}

pub fn multicast_add(mc: Handle) -> Result<(), String> {
    ok(unsafe { sys::cuMulticastAddDevice(mc, device()?) }, "cuMulticastAddDevice")
}

pub fn multicast_bind(mc: Handle, h: Handle, len: usize) -> Result<(), String> {
    ok(unsafe { sys::cuMulticastBindMem(mc, 0, h, 0, len, 0) }, "cuMulticastBindMem")
}
