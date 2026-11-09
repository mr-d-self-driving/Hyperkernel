use std::process::{Command, Output};

fn hyperserve(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_hyperserve"))
        .args(args)
        .env_remove("MG3_LINK")
        .output()
        .expect("hyperserve runs")
}

fn text(o: &Output) -> String {
    format!("{}{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr))
}

#[test]
fn each_task_is_a_subcommand_with_its_models_flags() {
    let tasks: [(&str, &[&str]); 4] = [
        (
            "llm",
            &[
                "--model",
                "--weights",
                "--kernels",
                "--preset",
                "--tokenizer",
                "--host",
                "--port",
                "--api-key",
                "--budget",
                "--pf-every",
                "--headroom",
            ],
        ),
        ("drive", &["--model", "--weights", "--kernels", "--frames", "--count", "--out"]),
        (
            "dna",
            &[
                "--model",
                "--weights",
                "--kernels",
                "--sequence",
                "--steps",
                "--top-k",
                "--top-p",
                "--temperature",
                "--seed",
                "--out",
            ],
        ),
        (
            "world",
            &[
                "--model",
                "--weights",
                "--kernels",
                "--steps",
                "--image-latent",
                "--context",
                "--noise",
                "--actions",
                "--frames-out",
                "--timing-out",
            ],
        ),
    ];
    for (task, flags) in tasks {
        let help = hyperserve(&[task, "--help"]);
        assert!(help.status.success(), "{task} --help: {}", text(&help));
        flags.iter().for_each(|f| assert!(text(&help).contains(f), "{task} --help has no {f}"));
        assert!(
            !text(&help).contains("--gpus") && !(task != "llm" && text(&help).contains("--preset")),
            "{}",
            text(&help)
        );
    }
}

#[test]
fn a_preset_is_one_of_the_llms() {
    let wrong = hyperserve(&["llm", "--model", "deepseek-v4.1-flash", "--weights", "w", "--preset", "c3"]);
    assert!(!wrong.status.success());
    assert!(text(&wrong).contains("c1, c2, c4, c5, c6, c7, c8"), "{}", text(&wrong));
}

#[test]
fn a_model_is_a_closed_set_per_task() {
    let wrong = hyperserve(&["dna", "--model", "openpilot", "--weights", "w", "--sequence", "s"]);
    assert!(!wrong.status.success());
    assert!(text(&wrong).contains("evo2-7b"), "{}", text(&wrong));
}

#[test]
fn a_task_runs_its_model_with_the_kernels_given() {
    let missing = "/nonexistent/hyperserve-sequence";
    let run = hyperserve(&[
        "dna",
        "--model",
        "evo2-7b",
        "--weights",
        "/nonexistent",
        "--kernels",
        "/nonexistent",
        "--sequence",
        missing,
    ]);
    assert!(!run.status.success());
    assert!(text(&run).starts_with(&format!("hyperserve: --sequence {missing}: ")), "{}", text(&run));
}
