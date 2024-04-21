package dev.vndx.mixin;

import net.minecraft.client.Minecraft;
import net.minecraft.client.gui.GuiMainMenu;
import net.minecraft.client.renderer.GlStateManager;
import net.minecraft.client.renderer.texture.DynamicTexture;
import net.minecraft.util.ResourceLocation;
import org.lwjgl.BufferUtils;
import org.lwjgl.opengl.GL11;
import org.lwjgl.opengl.GL15;
import org.lwjgl.opengl.GL20;
import org.lwjgl.opengl.GL30;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;

@Mixin(GuiMainMenu.class)
public class MixinGuiMainMenu {

    private static final float[] vertices = {
            1f, 1f, 1f, 0f,
            0f, 1f, 0f, 0f,
            0f, 0f, 0f, 1f,
            1f, 0f, 1f, 1f
    };
    private static final int[] indices = {
            0, 1, 2, 0, 2, 3
    };

    int vao = 0;
    int program = 0;
    int pos_loc = 0;
    int tc_loc = 0;

    @Inject(method = "initGui", at = @At("HEAD"))
    public void onInitGui(CallbackInfo ci) {
        try {
            InputStream vert_input_stream = Minecraft.getMinecraft().getResourceManager()
                    .getResource(new ResourceLocation("shaders:basic.vert")).getInputStream();
            InputStream frag_input_stream = Minecraft.getMinecraft().getResourceManager()
                    .getResource(new ResourceLocation("shaders:basic.frag")).getInputStream();

            ByteArrayOutputStream stream = new ByteArrayOutputStream();
            byte[] buf = new byte[1024];
            int read;

            while ((read = vert_input_stream.read(buf)) > 0) {
                stream.write(buf, 0, read);
            }

            byte[] vert_data = stream.toByteArray();

            stream.reset();

            while ((read = frag_input_stream.read(buf)) > 0) {
                stream.write(buf, 0, read);
            }

            byte[] frag_data = stream.toByteArray();

            ByteBuffer vert_buf = BufferUtils.createByteBuffer(vert_data.length);
            ByteBuffer frag_buf = BufferUtils.createByteBuffer(frag_data.length);

            vert_buf.put(vert_data);
            frag_buf.put(frag_data);
            vert_buf.flip();
            frag_buf.flip();

            int vert_shader = GL20.glCreateShader(GL20.GL_VERTEX_SHADER);
            int frag_shader = GL20.glCreateShader(GL20.GL_FRAGMENT_SHADER);

            GL20.glShaderSource(vert_shader, vert_buf);
            GL20.glShaderSource(frag_shader, frag_buf);

            GL20.glCompileShader(vert_shader);
            GL20.glCompileShader(frag_shader);

            program = GL20.glCreateProgram();

            GL20.glAttachShader(program, vert_shader);
            GL20.glAttachShader(program, frag_shader);

            GL20.glLinkProgram(program);

            GL20.glUseProgram(program);

            pos_loc = GL20.glGetAttribLocation(program, "pos");
            tc_loc = GL20.glGetAttribLocation(program, "tex_coords");

            GL20.glDeleteShader(vert_shader);
            GL20.glDeleteShader(frag_shader);
        } catch (IOException e) {
            System.out.println("Couldn't load shaders");
            e.printStackTrace();
        }

        // Data
        FloatBuffer vertices_buf = BufferUtils.createFloatBuffer(vertices.length);
        IntBuffer indices_buf = BufferUtils.createIntBuffer(indices.length);

        vertices_buf.put(vertices);
        indices_buf.put(indices);

        vertices_buf.rewind();
        indices_buf.rewind();

        // Get a VAO and two VBOs
        vao = GL30.glGenVertexArrays();
        int vertex_vbo = GL15.glGenBuffers();
        int index_vbo = GL15.glGenBuffers();

        GL30.glBindVertexArray(vao);
        GL15.glBindBuffer(GL15.GL_ARRAY_BUFFER, vertex_vbo);
        GL15.glBindBuffer(GL15.GL_ELEMENT_ARRAY_BUFFER, index_vbo);

        // Populate VBOs
        GL15.glBufferData(GL15.GL_ARRAY_BUFFER, vertices_buf, GL15.GL_STATIC_DRAW);
        GL15.glBufferData(GL15.GL_ELEMENT_ARRAY_BUFFER, indices_buf, GL15.GL_STATIC_DRAW);

        // Populate VAO
        GL20.glEnableVertexAttribArray(pos_loc);
        GL20.glEnableVertexAttribArray(tc_loc);
        GL20.glVertexAttribPointer(pos_loc, 2, GL11.GL_FLOAT, false, 16, 0);
        GL20.glVertexAttribPointer(tc_loc, 2, GL11.GL_FLOAT, false, 16, 2 * 4);

        // Unbind because 1.8 was codded by apes
        GL30.glBindVertexArray(0);

        GL15.glBindBuffer(GL15.GL_ELEMENT_ARRAY_BUFFER, 0);
        GL15.glBindBuffer(GL15.GL_ARRAY_BUFFER, 0);

        GL20.glUseProgram(0);
    }

    @Inject(method = "drawScreen", at = @At("RETURN"))
    public void onDrawScreen(CallbackInfo ci) {
        GL20.glUseProgram(program);
        GL30.glBindVertexArray(vao);
        GL11.glDrawElements(GL11.GL_TRIANGLES, indices.length, GL11.GL_UNSIGNED_INT, 0);
        GL30.glBindVertexArray(0);
        GL20.glUseProgram(0); // Go back to fixed function
    }
}
