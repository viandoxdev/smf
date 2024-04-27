package dev.vndx

import dev.vndx.bindings.*
import dev.vndx.utils.GLExtra
import net.minecraft.client.Minecraft
import net.minecraft.client.resources.IResource
import net.minecraft.util.ResourceLocation
import org.apache.logging.log4j.Logger
import org.lwjgl.BufferUtils
import org.lwjgl.opengl.*
import org.lwjgl.util.glu.GLU
import java.io.IOException
import java.nio.ByteBuffer

data class Texture(val size: Int, val channelCount: Int, val obj: Int)

data class Mesh(val vao: Int, val tex: Int, val indices_len: Long)

fun handle_errors(logger: Logger, s: String) {
    var err: Int;
    while(GL11.glGetError().also { err = it } != GL11.GL_NO_ERROR) {
        logger.error("[$s] opengl error: ${GLU.gluErrorString(err)}")
    }
}

class DrawString(val logger: Logger) {
    companion object {
        const val POS_LOC = 0
        const val TC_LOC = 1
    }

    var program = 0
    val meshes = ArrayList<Mesh>()
    val textures = ArrayList<Texture?>()

     init {
         try {
             val vertData = Minecraft.getMinecraft().resourceManager
                 .getResource(ResourceLocation("shaders:basic.vert")).inputStream.readBytes()
             val fragData = Minecraft.getMinecraft().resourceManager
                 .getResource(ResourceLocation("shaders:basic.frag")).inputStream.readBytes()

             val vertBuf = BufferUtils.createByteBuffer(vertData.size)
             val fragBuf = BufferUtils.createByteBuffer(fragData.size)

             vertBuf.put(vertData)
             fragBuf.put(fragData)
             vertBuf.flip()
             fragBuf.flip()

             val vertShader = GL20.glCreateShader(GL20.GL_VERTEX_SHADER)
             val fragShader = GL20.glCreateShader(GL20.GL_FRAGMENT_SHADER)

             GL20.glShaderSource(vertShader, vertBuf)
             GL20.glShaderSource(fragShader, fragBuf)

             logger.info("Loaded shaders")

             GL20.glCompileShader(vertShader)
             GL20.glCompileShader(fragShader)

             if(GL20.glGetShaderi(vertShader, GL20.GL_COMPILE_STATUS) != GL11.GL_TRUE) {
                 val log = GL20.glGetShaderInfoLog(vertShader, 1024)
                 logger.error("Vertex shader failed to compile with log: $log")
             }

             if(GL20.glGetShaderi(fragShader, GL20.GL_COMPILE_STATUS) != GL11.GL_TRUE) {
                 val log = GL20.glGetShaderInfoLog(fragShader, 1024)
                 logger.error("Fragment shader failed to compile with log: $log")
             }

             logger.info("Compiled shaders")

             program = GL20.glCreateProgram()

             GL20.glAttachShader(program, vertShader)
             GL20.glAttachShader(program, fragShader)

             GL20.glBindAttribLocation(program, POS_LOC, "pos")
             GL20.glBindAttribLocation(program, TC_LOC, "tex_coords")

             GL20.glLinkProgram(program)

             GL20.glUseProgram(program)

             GL20.glDeleteShader(vertShader)
             GL20.glDeleteShader(fragShader)

             logger.info("Linked shaders")

             handle_errors(logger, "SHADERS")

         } catch (e: IOException) {
             logger.error("Couldn't load shaders: $e")
         }

         try {
             val resource: IResource = Minecraft.getMinecraft().resourceManager
                 .getResource(ResourceLocation("fonts:font.ttc"))
             val bytes = resource.inputStream.readBytes()

             val buffer = ByteBuffer.allocateDirect(bytes.size)
             buffer.put(bytes)

             logger.info("Got direct byte buffer to font data: ${buffer.capacity()}")

             val config = FontConfig(rasterKind = RasterKind.Bitmap, scale = 1f / 4f, rasterScale = 1f / 2f, lineHeight = 1f, language = "en")

             val fonts = Font.load(buffer, config)

             val font = fonts[0]

             val res = font.processBatched(arrayOf(Command("This is ass صحبت حت latin text", true, 1f)))

             for(diff in res.diffs) {

                 val format = when(diff.channelCount) {
                     1 -> GL11.GL_RED
                     else -> GL11.GL_RGBA
                 }

                 val data_format = when(diff.channelCount) {
                     1 -> GL11.GL_RED
                     else -> GL11.GL_RGB
                 }

                 when(diff) {
                     is Diff.TextureCreation -> {
                         // Add as many nulls as needed to make room for the texture (usually just one)
                         textures.addAll((textures.size..diff.tex).map { null })

                         val obj = GL11.glGenTextures()

                         val bound = GL11.glGetInteger(GL11.GL_TEXTURE_BINDING_2D)


                         GL11.glBindTexture(GL11.GL_TEXTURE_2D, obj)

                         GL11.glTexParameteri(GL11.GL_TEXTURE_2D, GL11.GL_TEXTURE_WRAP_S, GL12.GL_CLAMP_TO_EDGE)
                         GL11.glTexParameteri(GL11.GL_TEXTURE_2D, GL11.GL_TEXTURE_WRAP_T, GL12.GL_CLAMP_TO_EDGE)
                         GL11.glTexParameteri(GL11.GL_TEXTURE_2D, GL11.GL_TEXTURE_MIN_FILTER, GL11.GL_LINEAR)
                         GL11.glTexParameteri(GL11.GL_TEXTURE_2D, GL11.GL_TEXTURE_MAG_FILTER, GL11.GL_LINEAR)
                         GLExtra.glTexImage2D(GL11.GL_TEXTURE_2D, 0, format, diff.size, diff.size, 0, data_format, GL11.GL_UNSIGNED_BYTE, diff.data)

                         GL11.glBindTexture(GL11.GL_TEXTURE_2D, bound)

                         textures[diff.tex] = Texture(diff.size, diff.channelCount, obj)
                     }
                     is Diff.TextureUpdate -> {
                         val tex = textures[diff.tex] as Texture

                         val bound = GL11.glGetInteger(GL11.GL_TEXTURE_BINDING_2D)
                         GL11.glBindTexture(GL11.GL_TEXTURE_2D, tex.obj)

                         val width = diff.damageMaxX - diff.damageMinX
                         val height = diff.damageMaxY - diff.damageMinY

                         GL11.glPixelStorei(GL11.GL_UNPACK_ROW_LENGTH, diff.size)
                         GL11.glPixelStorei(GL11.GL_UNPACK_SKIP_PIXELS, diff.damageMinX)
                         GL11.glPixelStorei(GL11.GL_UNPACK_SKIP_ROWS, diff.damageMinY)
                         GLExtra.glTexSubImage2D(GL11.GL_TEXTURE_2D, 0, diff.damageMinX, diff.damageMinY, width, height, data_format, GL11.GL_UNSIGNED_BYTE, diff.data)

                         GL11.glBindTexture(GL11.GL_TEXTURE_2D, bound)
                     }
                 }
             }

             handle_errors(logger, "TEXTURES")

             // We only have 1 command, so we only get one mesh array
             meshes.addAll(res.meshes[0].map {
                 // Get a VAO and two VBOs
                 val vao = GL30.glGenVertexArrays()
                 val vertexVbo = GL15.glGenBuffers()
                 val indexVbo = GL15.glGenBuffers()

                 GL30.glBindVertexArray(vao)
                 GL15.glBindBuffer(GL15.GL_ARRAY_BUFFER, vertexVbo)
                 GL15.glBindBuffer(GL15.GL_ELEMENT_ARRAY_BUFFER, indexVbo)

                 // Populate VBOs
                 GLExtra.glBufferData(GL15.GL_ARRAY_BUFFER, it.vertices_addr, it.vertices_len * 4, GL15.GL_STATIC_DRAW)
                 GLExtra.glBufferData(GL15.GL_ELEMENT_ARRAY_BUFFER, it.indices_addr, it.indices_len * 4, GL15.GL_STATIC_DRAW)

                 // Populate VAO
                 GL20.glEnableVertexAttribArray(POS_LOC)
                 GL20.glEnableVertexAttribArray(TC_LOC)
                 GL20.glVertexAttribPointer(POS_LOC, 2, GL11.GL_FLOAT, false, 16, 0)
                 GL20.glVertexAttribPointer(TC_LOC, 2, GL11.GL_FLOAT, false, 16, 8)

                 Mesh(vao, it.tex, it.indices_len)
             })

             handle_errors(logger, "MESH")

             // Unbind because 1.8 was codded by apes
             GL30.glBindVertexArray(0)

             GL15.glBindBuffer(GL15.GL_ELEMENT_ARRAY_BUFFER, 0)
             GL15.glBindBuffer(GL15.GL_ARRAY_BUFFER, 0)

             GL20.glUseProgram(0)

             handle_errors(logger, "FINISH")
         } catch (e: IOException) {
             logger.error("Couldn't load fonts $e")
         }
     }

    fun draw() {
        GL20.glUseProgram(program)
        for(mesh in meshes) {
            GL30.glBindVertexArray(mesh.vao)
            GL11.glBindTexture(GL11.GL_TEXTURE_2D, textures[mesh.tex]!!.obj)
            GL11.glDrawElements(GL11.GL_TRIANGLES, mesh.indices_len.toInt(), GL11.GL_UNSIGNED_INT, 0)
        }
        GL30.glBindVertexArray(0)
        GL20.glUseProgram(0)
    }
}