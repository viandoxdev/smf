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

        try {
            val resource: net.minecraft.client.resources.IResource = Minecraft.getMinecraft().getResourceManager()
                .getResource(net.minecraft.util.ResourceLocation("fonts:font.ttc"))
            val bytes = resource.inputStream.readBytes()

            val buffer = ByteBuffer.allocateDirect(bytes.size)
            buffer.put(bytes)

            logger.info("Got direct byte buffer to font data: ${buffer.capacity()}")

            val config = FontConfig(rasterKind = RasterKind.Bitmap, scale = 1.0f, lineHeight = 1.5f)

            val fonts = Font.load(buffer, config)

            for (font in fonts) {
                logger.info("Loaded font '${font.name} ${font.config}'")
            }
        } catch (e: java.io.IOException) {
            throw java.lang.RuntimeException(e)
        }
    }
}
