package dev.vndx.bindings

import dev.vndx.SMF
import net.minecraft.client.Minecraft
import org.apache.logging.log4j.Logger
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.IOException
import java.nio.file.Files
import java.security.MessageDigest
import java.util.Base64
import java.util.Objects
import java.util.Scanner


class UnsupportedPlatformException(s: String) : RuntimeException(s)


data class Metadata(val size: Long, val sha256Base64: String)

fun getPlatform(): String {
    //TODO: override from config
    val osName = System.getProperty("os.name")
    val osArch = System.getProperty("os.arch")

    val resOs = when {
        osName.startsWith("Windows") -> "pc-windows-gnu"
        osName.startsWith("Linux") -> "unknown-linux-gnu"
        osName.startsWith("FreeBSD") -> "unknown-freebsd"
        osName.startsWith("MAC OS X") || osName.startsWith("Darwin") -> "apple-darwin"
        else -> throw UnsupportedPlatformException("Cannot determine Rust platform name: unrecognized OS name $osName")
    }

    val resArch = when (osArch) {
        "x86_64", "amd64" -> "x86_64"
        "x86", "i686" -> "i686"
        "aarch64" -> "aarch64"
        else -> throw  UnsupportedPlatformException("Cannot determine Rust platform name: unrecognized architecture name $osArch")
    }

    return "$resArch-$resOs"
}

fun readFromChecksumsTxt(soName: String, logger: Logger): Metadata? {
    try {
        SMF::class.java.getResourceAsStream("/natives/checksums.txt").use { inputStream ->
            val scanner = Scanner(Objects.requireNonNull(inputStream, "checksums.txt not found"))
            while (scanner.hasNextLine()) {
                val line: String = scanner.nextLine()
                if (line.isEmpty()) continue
                val comps =
                    line.split("\t".toRegex()).dropLastWhile { it.isEmpty() }.toTypedArray()
                if (comps.size != 3) throw IOException("Malformed line in checksums.txt: $line")
                if (comps[0] == soName) {
                    return Metadata(comps[1].toLong(), comps[2])
                }
            }
        }
    } catch (e: IOException) {
        logger.error("Could not read checksums.txt", e)
    }

    return null
}

fun shouldRewrite(dest: File, metadata: Metadata?, logger: Logger): Boolean {
    try {
        return when {
            metadata == null -> true
            !dest.exists() -> true
            Files.size(dest.toPath()) != metadata.size -> true
            else -> {
                val digest = MessageDigest.getInstance("SHA-256")
                FileInputStream(dest).use {
                    val bytes = ByteArray(4096)
                    while (true) {
                        val count = it.read(bytes)
                        if (count == 0) {
                            break
                        }
                        digest.update(bytes.sliceArray(0 until count))
                    }
                }
                val digestStr = Base64.getEncoder().encodeToString(digest.digest())
                return !digestStr.equals(metadata.sha256Base64)
            }
        }
    } catch (e: IOException) {
        logger.error("Could not determine whether we need to overwrite existing file in .minecraft; assuming we do", e)
        return true
    }
}

fun loadNativeLibrary(logger: Logger) {
    val dir = Minecraft.getMinecraft().mcDataDir
    val soName = System.mapLibraryName("smf")
    val soNameWithPlatform = "${getPlatform()}-$soName"
    val metadata = readFromChecksumsTxt(soName, logger)

    try {
        SMF::class.java.getResourceAsStream("/natives/$soNameWithPlatform").use {
            if (it === null) {
                throw UnsupportedPlatformException("Could not find $soNameWithPlatform")
            }

            val tmp = File(dir, soName)

            if(shouldRewrite(tmp, metadata, logger)) {
                FileOutputStream(tmp).use { output ->
                    it.copyTo(output)
                }
            } else {
                logger.info("Native library already copied")
            }

            System.load(tmp.absolutePath)
        }
    } catch (e: Exception) {
        logger.error("Failed to load library", e)
        //loadingException = e;
    }

}