package dev.vndx

import dev.vndx.bindings.loadNativeLibrary
import dev.vndx.bindings.test_jni
import net.minecraftforge.fml.common.Mod
import net.minecraftforge.fml.common.event.FMLInitializationEvent
import org.apache.logging.log4j.LogManager

@Mod(modid = "smf", useMetadata = true)
class SMF {
    @Mod.EventHandler
    fun init(event: FMLInitializationEvent) {

        val logger = LogManager.getLogger()

        loadNativeLibrary(logger)

        logger.info("JNI: ${test_jni("Test String from Kotlin")}")
    }
}
