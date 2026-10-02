using System.Diagnostics;
using System.Security.Cryptography;

namespace Nam2ZoomDesktop;

internal static class PortableRuntime
{
    public static bool IsPortable(string root) => File.Exists(Path.Combine(root, "runtime", "portable.marker"));

    public static bool VisualCppReady()
    {
        var dll = Path.Combine(Environment.SystemDirectory, "msvcp140.dll");
        if (!File.Exists(dll)) return false;
        var version = FileVersionInfo.GetVersionInfo(dll);
        return version.FileMajorPart > 14 || version.FileMajorPart == 14 && version.FileMinorPart >= 44;
    }

    public static async Task EnsureVisualCppAsync(string root, Action<string> report,
        CancellationToken token)
    {
        if (VisualCppReady()) return;
        var directory = Path.Combine(root, ".tooling", "prerequisites");
        Directory.CreateDirectory(directory);
        var installer = Path.Combine(directory, "vc_redist.x64.exe");
        report("Downloading Microsoft Visual C++ x64 runtime...");
        using var client = new HttpClient();
        var data = await client.GetByteArrayAsync("https://aka.ms/vs/17/release/vc_redist.x64.exe", token);
        await File.WriteAllBytesAsync(installer, data, token);
        // Windows verifies the file signature, not just the certificate name.
        var literal = installer.Replace("'", "''");
        var script = "$s=Get-AuthenticodeSignature -LiteralPath '" + literal
            + "'; if($s.Status -eq 'Valid' -and $s.SignerCertificate.Subject -match '(?:^|,\\s*)CN=Microsoft Corporation(?:,|$)'){exit 0}else{exit 1}";
        var encoded = Convert.ToBase64String(System.Text.Encoding.Unicode.GetBytes(script));
        var powershell = Path.Combine(Environment.SystemDirectory, "WindowsPowerShell", "v1.0", "powershell.exe");
        await RunAsync(powershell, root, ["-NoProfile", "-NonInteractive", "-EncodedCommand", encoded], report, token);
        token.ThrowIfCancellationRequested();
        report("Installing Microsoft runtime; approve the Windows administrator prompt for Microsoft Corporation.");
        using var process = Process.Start(new ProcessStartInfo(installer) {
            UseShellExecute = true, Verb = "runas", Arguments = "/install /passive /norestart"
        }) ?? throw new InvalidOperationException("Could not start Microsoft runtime installer");
        await process.WaitForExitAsync();
        if (process.ExitCode is not (0 or 3010) || !VisualCppReady())
            throw new InvalidOperationException($"Microsoft runtime setup did not complete (exit {process.ExitCode}). Check its result before retrying.");
        if (process.ExitCode == 3010)
            throw new InvalidOperationException("Microsoft runtime installed but requires a Windows restart. Restart before converting or connecting the pedal.");
    }

    public static void Prepare(string root)
    {
        if (!IsPortable(root)) return;
        var config = Path.Combine(root, ".tooling", "nam-train-venv", "pyvenv.cfg");
        if (!File.Exists(config)) return;
        UpdateVenv(config, Path.Combine(root, "runtime", "python312"));
    }

    private static void UpdateVenv(string config, string pythonHome)
    {
        var executable = Path.Combine(pythonHome, "python.exe");
        if (!File.Exists(executable) || !File.Exists(config))
            throw new FileNotFoundException("Portable Python runtime is incomplete", config);

        var lines = File.ReadAllLines(config);
        var changed = false;
        for (var i = 0; i < lines.Length; i++) {
            var separator = lines[i].IndexOf('=');
            if (separator < 0) continue;
            var key = lines[i][..separator].Trim();
            var value = key.Equals("home", StringComparison.OrdinalIgnoreCase) ? pythonHome :
                key.Equals("executable", StringComparison.OrdinalIgnoreCase) ? executable : null;
            if (value is null || lines[i][(separator + 1)..].Trim() == value) continue;
            lines[i] = $"{key} = {value}";
            changed = true;
        }
        if (changed) File.WriteAllLines(config, lines);
    }

    private static string TrainingWheel(string root)
    {
        var runtime = Path.Combine(root, "runtime");
        var wheels = Directory.Exists(runtime)
            ? Directory.GetFiles(runtime, "neural_amp_modeler-*.whl") : [];
        if (wheels.Length != 1)
            throw new InvalidOperationException(wheels.Length == 0
                ? "The portable training wheel is missing. Extract the entire corrected release ZIP to a new folder and retry."
                : "The portable runtime contains multiple training wheels. Extract the release ZIP to a new folder and retry.");
        return wheels[0];
    }

    private static string Fingerprint(string root)
    {
        var wheel = TrainingWheel(root);
        var constraints = Path.Combine(root, "runtime", "training-constraints.txt");
        return Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(wheel)))
             + Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(constraints)));
    }

    public static bool TrainingReady(string root)
    {
        var marker = Path.Combine(root, ".tooling", "training.ready");
        return File.Exists(marker) && File.ReadAllText(marker).Trim() == Fingerprint(root)
            && File.Exists(Path.Combine(root, ".tooling", "nam-train-venv", "Scripts", "python.exe"));
    }

    public static async Task EnsureTrainingAsync(string root, bool gpu, Action<string> report,
        CancellationToken cancellationToken)
    {
        if (TrainingReady(root)) return;
        var wheel = TrainingWheel(root);
        var python = Path.Combine(root, "runtime", "python312", "python.exe");
        var venv = Path.Combine(root, ".tooling", "nam-train-venv");
        Directory.CreateDirectory(Path.Combine(root, ".tooling"));
        await RunAsync(python, root, ["-m", "venv", venv], report, cancellationToken);
        var trainer = Path.Combine(venv, "Scripts", "python.exe");
        var index = gpu ? "https://download.pytorch.org/whl/cu128" : "https://download.pytorch.org/whl/cpu";
        await RunAsync(trainer, root, ["-m", "pip", "install", "--only-binary=:all:",
            "--force-reinstall", "--constraint", Path.Combine(root, "runtime", "training-constraints.txt"),
            "torch==2.11.0", "--index-url", index], report, cancellationToken);
        await RunAsync(trainer, root, ["-m", "pip", "install", "--only-binary=:all:",
            "--constraint", Path.Combine(root, "runtime", "training-constraints.txt"),
            "--index-url", "https://pypi.org/simple", wheel, "soundfile==0.14.0"], report, cancellationToken);
        await RunAsync(trainer, root, ["-m", "pip", "install", "--no-deps", "--force-reinstall", wheel], report, cancellationToken);
        await RunAsync(trainer, root, ["-c", "import nam.cli, torch, scipy, soundfile; print('Training runtime verified; CUDA available:', torch.cuda.is_available())"], report, cancellationToken);
        cancellationToken.ThrowIfCancellationRequested();
        File.WriteAllText(Path.Combine(root, ".tooling", "training.ready"), Fingerprint(root));
    }

    private static async Task RunAsync(string python, string root, string[] args,
        Action<string> report, CancellationToken token)
    {
        token.ThrowIfCancellationRequested();
        var start = new ProcessStartInfo(python) {
            WorkingDirectory = root, UseShellExecute = false, CreateNoWindow = true,
            RedirectStandardOutput = true, RedirectStandardError = true
        };
        start.Environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1";
        start.Environment["PIP_CACHE_DIR"] = Path.Combine(root, ".tooling", "pip-cache");
        start.Environment["PIP_CONFIG_FILE"] = "NUL";
        start.Environment.Remove("PIP_EXTRA_INDEX_URL");
        start.Environment["PYTHONUNBUFFERED"] = "1";
        foreach (var arg in args) start.ArgumentList.Add(arg);
        using var process = Process.Start(start) ?? throw new InvalidOperationException("Could not start runtime setup");
        using var cancellation = token.Register(() => {
            try { if (!process.HasExited) process.Kill(entireProcessTree: true); }
            catch (InvalidOperationException) { }
            catch (System.ComponentModel.Win32Exception) { }
        });
        async Task Drain(StreamReader reader)
        {
            string? line;
            while ((line = await reader.ReadLineAsync()) is not null) report(line);
        }
        var stdout = Drain(process.StandardOutput);
        var stderr = Drain(process.StandardError);
        await process.WaitForExitAsync();
        await Task.WhenAll(stdout, stderr);
        token.ThrowIfCancellationRequested();
        if (process.ExitCode != 0) throw new InvalidOperationException($"Dependency setup failed (exit {process.ExitCode}). Check the log and internet connection before retrying.");
    }
}
