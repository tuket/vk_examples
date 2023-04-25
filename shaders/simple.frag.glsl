#version 450
#pragma shader_stage(fragment)

layout(location = 0) out vec4 o_color;

layout(location = 0) in vec2 v_texCoord;

layout(set = 1, binding = 1) uniform sampler2D u_texture;

void main()
{
    o_color = texture(u_texture, v_texCoord);
    //o_color = vec4(1, 0, 0, 1);
}