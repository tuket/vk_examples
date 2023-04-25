#version 450
#pragma shader_stage(vertex)

layout(location = 0) in vec2 a_pos;
layout(location = 1) in vec2 a_texCoord;

// redefine the default gl_PerVertex: https://www.khronos.org/opengl/wiki/Built-in_Variable_(GLSL)#Vertex_shader_outputs
out gl_PerVertex
{
    vec4 gl_Position;
};
layout(location = 0) out vec2 v_texCoord;

layout(set = 0, binding = 0) uniform CameraParams {
    vec2 u_camTranslation;
    vec2 u_camRotScale_X;
    vec2 u_camRotScale_Y;
};

layout(set = 1, binding = 0) uniform ObjectParams {
    vec2 u_objTranslation;
    vec2 u_objRotScale_X;
    vec2 u_objRotScale_Y;
};

void main()
{
    vec2 worldSpacePos = mat2(u_objRotScale_X, u_objRotScale_Y) * a_pos + u_objTranslation;
    vec2 cameraSpacePos = transpose(mat2(u_camRotScale_X, u_camRotScale_Y)) * worldSpacePos - u_camTranslation;
    gl_Position = vec4(cameraSpacePos, 0, 1);
    v_texCoord = a_texCoord;
}