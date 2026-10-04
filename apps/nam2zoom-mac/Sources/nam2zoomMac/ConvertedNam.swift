import Foundation

/// Mirrors apps/nam2zoom-desktop/ConvertedNam.cs: exports each adapted
/// (already pedal-shaped) NAM under its original filename so it can be kept,
/// shared, or added to a later bank without retraining. An identical export
/// is reused; a different conversion with the same name gets " (2)", " (3)"
/// ...; existing files are never overwritten.
enum ConvertedNam {
    /// "Converted_NAM" beside the app bundle, like the Windows app puts it
    /// beside the EXE. Falls back to Application Support when that location
    /// isn't writable (e.g. an app installed in /Applications).
    static func directory(fallback: URL) -> URL {
        let beside = Bundle.main.bundleURL
            .deletingLastPathComponent()
            .appendingPathComponent("Converted_NAM", isDirectory: true)
        let fm = FileManager.default
        do {
            try fm.createDirectory(at: beside, withIntermediateDirectories: true)
            if fm.isWritableFile(atPath: beside.path) { return beside }
        } catch {}
        return fallback.appendingPathComponent("Converted_NAM", isDirectory: true)
    }

    static func save(student: URL, original: URL, into directory: URL) throws -> URL {
        let fm = FileManager.default
        try fm.createDirectory(at: directory, withIntermediateDirectories: true)
        let name = original.lastPathComponent
        let stem = original.deletingPathExtension().lastPathComponent
        let ext = original.pathExtension
        let bytes = try Data(contentsOf: student)

        var number = 1
        while true {
            let candidate = number == 1
                ? name
                : (ext.isEmpty ? "\(stem) (\(number))" : "\(stem) (\(number)).\(ext)")
            let destination = directory.appendingPathComponent(candidate)
            if fm.fileExists(atPath: destination.path) {
                if try Data(contentsOf: destination) == bytes { return destination }
                number += 1
                continue
            }
            // Non-atomic create-new: never replaces an existing file.
            try bytes.write(to: destination, options: .withoutOverwriting)
            return destination
        }
    }
}
