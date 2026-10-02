using Nam2ZoomDesktop;

if (args is ["--self-test"]) {
    var temporary = Path.Combine(Path.GetTempPath(), "nam2zoom-runtime-" + Guid.NewGuid());
    var runtime = Path.Combine(temporary, "runtime");
    Directory.CreateDirectory(runtime);
    try {
        async Task ExpectFailure(string expected)
        {
            try {
                await PortableRuntime.EnsureTrainingAsync(temporary, false,
                    _ => throw new Exception("Unexpected dependency setup"), CancellationToken.None);
            } catch (InvalidOperationException ex) when (ex.Message.Contains(expected)) {
                return;
            }
            throw new Exception("Expected runtime validation failure: " + expected);
        }
        await ExpectFailure("training wheel is missing");
        File.WriteAllText(Path.Combine(runtime, "wheel-copy-probe.whl"), "test");
        await ExpectFailure("training wheel is missing");
        var wheel = Path.Combine(runtime, "neural_amp_modeler-0.1.dev1-py3-none-any.whl");
        File.WriteAllText(wheel, "test");
        File.WriteAllText(Path.Combine(runtime, "neural_amp_modeler-0.2-py3-none-any.whl"), "test");
        await ExpectFailure("multiple training wheels");
        File.Delete(Path.Combine(runtime, "neural_amp_modeler-0.2-py3-none-any.whl"));
        var constraints = Path.Combine(runtime, "training-constraints.txt");
        File.WriteAllText(constraints, "test constraints");
        Directory.CreateDirectory(Path.Combine(temporary, ".tooling", "nam-train-venv", "Scripts"));
        File.WriteAllText(Path.Combine(temporary, ".tooling", "nam-train-venv", "Scripts", "python.exe"), "test");
        var fingerprint = Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(File.ReadAllBytes(wheel)))
            + Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(File.ReadAllBytes(constraints)));
        File.WriteAllText(Path.Combine(temporary, ".tooling", "training.ready"), fingerprint);
        if (!PortableRuntime.TrainingReady(temporary)) throw new Exception("Valid runtime was not recognized");
        File.WriteAllText(wheel, "changed");
        if (PortableRuntime.TrainingReady(temporary)) throw new Exception("Changed wheel was not detected");
        Console.WriteLine("Portable runtime validation tests passed");
        return 0;
    } finally {
        Directory.Delete(temporary, recursive: true);
    }
}

if (args.Length != 2 || args[1] is not ("--check" or "--setup-cpu")) {
    Console.Error.WriteLine("Usage: RuntimeSmoke <extracted portable folder> --check|--setup-cpu");
    return 2;
}
var root = Path.GetFullPath(args[0]);
if (!PortableRuntime.IsPortable(root)) throw new InvalidOperationException("Not a portable release folder");
PortableRuntime.Prepare(root);
Console.WriteLine($"VisualCppReady: {PortableRuntime.VisualCppReady()}");
if (!PortableRuntime.VisualCppReady()) throw new InvalidOperationException("Microsoft C++ runtime prerequisite missing; test the GUI's signed-installer flow before continuing");
if (args[1] == "--setup-cpu")
    await PortableRuntime.EnsureTrainingAsync(root, false, Console.WriteLine, CancellationToken.None);
Console.WriteLine($"TrainingReady: {PortableRuntime.TrainingReady(root)}");
return PortableRuntime.TrainingReady(root) ? 0 : 1;
