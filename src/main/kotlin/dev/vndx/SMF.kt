package dev.vndx

import dev.vndx.bindings.*
import net.minecraft.client.Minecraft
import net.minecraftforge.fml.common.Mod
import net.minecraftforge.fml.common.event.FMLInitializationEvent
import org.apache.logging.log4j.LogManager
import java.nio.ByteBuffer

@Mod(modid = "smf", useMetadata = true)
class SMF {
    @Mod.EventHandler
    fun init(event: FMLInitializationEvent) {
        val logger = LogManager.getLogger()
        loadNativeLibrary(logger)
        GlobalConfig(atlasSize = 4096, glyphPadding = 8, sinAlpha = 0.03, coloringSeed = 6942012345678980085).use()
    }
}
