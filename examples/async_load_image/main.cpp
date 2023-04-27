#include <stdio.h>
#include "../common.hpp"
#include <GLFW/glfw3.h>
#include <stb_image.h>
#include <thread>
#include <mutex>
#include <condition_variable>

#define MY_VULKAN_VERSION VK_API_VERSION_1_1
#define ENABLE_DEBUG_MESSENGER
#define SHADERS_FOLDER "shaders/"

struct Vert {
	vec2 pos;
	vec2 texCoord;
};

struct Buffer {
	VkBuffer buffer;
	VmaAllocation alloc;
	VmaAllocationInfo allocInfo;
};

struct Image {
	VkImage img;
	VkImageView view;
	VmaAllocation alloc;
	VmaAllocationInfo allocInfo;
};

struct CameraParams {
	vec2 translation;
	vec2 scaleRot_X;
	vec2 scaleRot_Y;
};

struct ObjectParams {
	vec2 translation;
	vec2 scaleRot_X;
	vec2 scaleRot_Y;
};

typedef void (*LoadImageCallback)(void* userPtr, CStr fileName, const Image& img);
struct LoadTask {
	CStr fileName;
	LoadImageCallback callback;
	void* callback_userPtr;
};
struct QueueTransferTask {
	CStr fileName;
	Image img;
	LoadImageCallback callback;
	void* callback_userPtr;
};

struct Vk {
	GLFWwindow* window = nullptr;
	VkInstance instance;
	VkSurfaceKHR surface;

	VkPhysicalDevice physicalDevice;
	VkPhysicalDeviceProperties physicalDeviceProps;
	VkPhysicalDeviceMemoryProperties physicalDeviceMemProps;
	VkDevice device;

	u32 queueFamily_main;
	u32 queueFamily_transfer;
	VkQueue queue_main;
	VkQueue queue_transfer;
	
	VmaAllocator allocator;

	vkh::Swapchain swapchain;
	VkFramebuffer framebuffers[vkh::Swapchain::MAX_IMAGES];

	VkSemaphore semaphore_swapchainImageAcquired[vkh::Swapchain::MAX_IMAGES];
	VkSemaphore semaphore_drawFinished[vkh::Swapchain::MAX_IMAGES];
	VkFence fence_queueWorkFinished[vkh::Swapchain::MAX_IMAGES];

	VkRenderPass renderPass;
	VkDescriptorSetLayout descSetLayout_camera;
	VkDescriptorSetLayout descSetLayout_object;
	VkPipelineLayout pipelineLayout;
	VkPipeline pipeline;

	VkCommandPool cmdPool_main;
	VkCommandPool cmdPool_transfer;

	VkCommandBuffer cmdBuffers_draw[vkh::Swapchain::MAX_IMAGES];
	VkCommandBuffer cmdBuffer_load;

	Buffer stagingBuffer;
	VkSampler sampler;
	Image blackImg;

	Buffer uniformBuffer_cameraParams;
	Buffer uniformBuffer_objectParams;

	VkDescriptorPool descPool;
	VkDescriptorSet descSet_cameraParams;
	VkDescriptorSet descSet_objectParams;

	struct Quad {
		Buffer vertBuffer;
		Buffer indBuffer;
	} quad;

	std::thread loaderThread;
	bool terminateLoaderThread = false;
	std::vector<LoadTask> loadTasks;
	std::mutex loadTasks_mutex;
	std::condition_variable loadTasks_condVar;
	std::vector<QueueTransferTask> queueTransferTasks;
	std::mutex queueTransferTasks_mutex;

	DelayedResourceDestructionManager<VkDescriptorSet> delayedDestrMan_descSet;

} vk;

static void onWindowResized(GLFWwindow* window, int w, int h)
{
	if (w == 0 || h == 0)
		return;
	vkDeviceWaitIdle(vk.device);
	vkh::create_swapChain(vk.swapchain, vk.physicalDevice, vk.device, vk.surface, 2, VK_PRESENT_MODE_FIFO_KHR);
	for (u32 i = 0; i < vk.swapchain.numImages; i++) {
		vkDestroyFramebuffer(vk.device, vk.framebuffers[i], nullptr);
		vk.framebuffers[i] = VK_NULL_HANDLE;
	}
}

static void recordCmds_copyBufferToImage(VkCommandBuffer cmdBuffer, VkImage image, VkBuffer buffer, u32 w, u32 h)
{
	const VkImageSubresourceRange subresRange = {
		.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
		.baseMipLevel = 0,
		.levelCount = 1,
		.baseArrayLayer = 0,
		.layerCount = 1,
	};

	// transition the image to the TRANSFER_DST_OPTIMAL layout
	const VkImageMemoryBarrier imgBarrier_0 = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
		.srcAccessMask = VK_ACCESS_NONE,
		.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
		.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
		.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		.image = image,
		.subresourceRange = subresRange,
	};
	vkCmdPipelineBarrier(cmdBuffer,
		VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
		0,
		0, nullptr,
		0, nullptr,
		1, &imgBarrier_0);

	// copy from the staging buffer to the image
	const VkBufferImageCopy region = {
		.bufferRowLength = w,
		.bufferImageHeight = h,
		.imageSubresource = {
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.mipLevel = 0,
			.baseArrayLayer = 0,
			.layerCount = 1,
		},
		.imageOffset = {0, 0, 0},
		.imageExtent = {w, h, 1},
	};
	vkCmdCopyBufferToImage(cmdBuffer, vk.stagingBuffer.buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
}

static void recordCmd_imageQueueFamilyTransfer(VkCommandBuffer cmdBuffer, VkImage img, u32 srcFamily, u32 dstFamily)
{
	const VkImageMemoryBarrier imgBarrier = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
		.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
		.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
		.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		.srcQueueFamilyIndex = srcFamily,
		.dstQueueFamilyIndex = dstFamily,
		.image = img,
		.subresourceRange = {
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.baseMipLevel = 0,
			.levelCount = 1,
			.baseArrayLayer = 0,
			.layerCount = 1,
		},
	};
	vkCmdPipelineBarrier(cmdBuffer,
		VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
		0,
		0, nullptr,
		0, nullptr,
		1, &imgBarrier
	);
}

static void recordCmd_transitionImageLayoutToShaderReadOnlyOptimal(VkCommandBuffer cmdBuffer, VkImage img)
{
	// transition the image to the SHADER_READ_ONLY_OPTIMAL layout
	const VkImageMemoryBarrier imgBarrier = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
		.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
		.dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
		.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		.image = img,
		.subresourceRange = {
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.baseMipLevel = 0,
			.levelCount = 1,
			.baseArrayLayer = 0,
			.layerCount = 1,
		},
	};
	vkCmdPipelineBarrier(cmdBuffer,
		VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
		0,
		0, nullptr,
		0, nullptr,
		1, &imgBarrier);
}

static void loaderThreadFn()
{
	VkCommandBuffer cmdBuffer;
	vkh::allocateCmdBuffers(vk.device, vk.cmdPool_transfer, { &cmdBuffer, 1 });
	const VkFence fence = vkh::createFence(vk.device, false);

	while (true)
	{
		std::unique_lock loadTasks_lock(vk.loadTasks_mutex);
		vk.loadTasks_condVar.wait(loadTasks_lock, [](){ return vk.loadTasks.size() > 0 || vk.terminateLoaderThread; });
		if (vk.terminateLoaderThread)
			break;

		auto task = vk.loadTasks.back();
		vk.loadTasks.pop_back();
		loadTasks_lock.unlock();

		int w, h, nc;
		u8* data = stbi_load(task.fileName, &w, &h, &nc, 4);
		// simulate slower load times
		//std::this_thread::sleep_for(std::chrono::milliseconds(1000));

		vkh::ImgInfo info = {
			.width = u32(w),
			.height = u32(h),
			.mipLevels = 1,
		};
		Image img;
		vkh::createImage_texture(vk.device, vk.allocator, info, img.img, img.alloc, &img.allocInfo, &img.view);

		assert(4 * w * h < vk.stagingBuffer.allocInfo.size);
		memcpy(vk.stagingBuffer.allocInfo.pMappedData, data, 4 * w * h);
		vmaFlushAllocation(vk.allocator, vk.stagingBuffer.alloc, 0, VK_WHOLE_SIZE);

		vkh::beginCmdBuffer(cmdBuffer, true);
		recordCmds_copyBufferToImage(cmdBuffer, img.img, vk.stagingBuffer.buffer, w, h);
		recordCmd_imageQueueFamilyTransfer(cmdBuffer, img.img, vk.queueFamily_transfer, vk.queueFamily_main);
		vkEndCommandBuffer(cmdBuffer);

		vkh::submit(vk.queue_transfer, { &cmdBuffer, 1 }, {}, {}, {}, fence);

		vkWaitForFences(vk.device, 1, &fence, VK_FALSE, -1);

		{
			std::lock_guard queueTransferTasks_lock(vk.queueTransferTasks_mutex);
			vk.queueTransferTasks.push_back({ task.fileName, img, task.callback, task.callback_userPtr });
		}
	}
}

static void loadImageAsync(CStr fileName, LoadImageCallback callback, void* callback_userPtr)
{
	std::lock_guard l(vk.loadTasks_mutex);
	vk.loadTasks.push_back({ fileName, callback, callback_userPtr });
	vk.loadTasks_condVar.notify_one();
}

static void recordCmds_doPendingQueueTransferTasks(VkCommandBuffer cmdBuffer)
{
	std::lock_guard l(vk.queueTransferTasks_mutex);
	if (vk.queueTransferTasks.empty())
		return;

	std::vector<VkImageMemoryBarrier> imgBarriers(vk.queueTransferTasks.size());
	for (size_t i = 0; i < vk.queueTransferTasks.size(); i++) {
		imgBarriers[i] = {
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
			.dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
			.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			.srcQueueFamilyIndex = vk.queueFamily_transfer,
			.dstQueueFamilyIndex = vk.queueFamily_main,
			.image = vk.queueTransferTasks[i].img.img,
			.subresourceRange = {
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1,
			},
		};
	}

	vkCmdPipelineBarrier(cmdBuffer,
		VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
		0,
		0, nullptr,
		0, nullptr,
		imgBarriers.size(), imgBarriers.data());

	for (const auto& task : vk.queueTransferTasks) {
		task.callback(task.callback_userPtr, task.fileName, task.img);
	}
	vk.queueTransferTasks.resize(0);
}

int main()
{
	int ok = glfwInit();
	assert(ok);

	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	vk.window = glfwCreateWindow(800, 600, "example", nullptr, nullptr);

	glfwSetFramebufferSizeCallback(vk.window, onWindowResized);

	VkResult vkRes;

	u32 maxVulkanVersion;
	vkEnumerateInstanceVersion(&maxVulkanVersion);
	const u32 vulkanVersion = glm::min(MY_VULKAN_VERSION, maxVulkanVersion);

	{ // create vulkan instance
		std::vector<CStr> instanceExtensions;
		u32 numExtensions;
		const char** extensionNames = glfwGetRequiredInstanceExtensions(&numExtensions);
		instanceExtensions.reserve(numExtensions + 8);
		for (u32 i = 0; i < numExtensions; i++)
			instanceExtensions.push_back(extensionNames[i]);

		#ifdef ENABLE_DEBUG_MESSENGER
			instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
		#endif

		vk.instance = vkh::createInstance(vulkanVersion, {}, instanceExtensions, "example");

		#ifdef ENABLE_DEBUG_MESSENGER
			const VkDebugUtilsMessengerCreateInfoEXT debugUtilsCreateInfo = {
				.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
				.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
				.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT,
				.pfnUserCallback = +[](
					VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
					VkDebugUtilsMessageTypeFlagsEXT message_type,
					const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
					void* user_data
				) -> VkBool32
				{
					printf("!!!\n");
					return VK_FALSE;
				},
			};
			VkDebugUtilsMessengerEXT messenger;
			auto vkCreateDebugUtilsMessenger = (decltype(&vkCreateDebugUtilsMessengerEXT))vkGetInstanceProcAddr(vk.instance, "vkCreateDebugUtilsMessengerEXT");
			vkRes = vkCreateDebugUtilsMessenger(vk.instance, &debugUtilsCreateInfo, nullptr, &messenger);
			vkh::assertRes(vkRes);
		#endif
	}

	// create window surface
	vkRes = glfwCreateWindowSurface(vk.instance, vk.window, nullptr, &vk.surface);
	vkh::assertRes(vkRes);

	vkh::findBestPhysicalDevice(vk.instance, vk.physicalDevice, vk.physicalDeviceProps, vk.physicalDeviceMemProps);
	vk.queueFamily_main = vkh::findGraphicsQueueFamily(vk.physicalDevice, vk.surface);
	vk.queueFamily_transfer = vkh::findTransferOnlyQueueFamily(vk.physicalDevice);
	if (vk.queueFamily_transfer < 0) {
		printf("No transfer-only queue family, trying with a different family...\n");

		// try to find alternative queue with TRANSFER capabilities
		[&]() {
			VkQueueFamilyProperties props[32];
			u32 numQueueFamilies = std::size(props);
			vkGetPhysicalDeviceQueueFamilyProperties(vk.physicalDevice, &numQueueFamilies, props);
			for (u32 i = 0; i < numQueueFamilies; i++) {
				if ((props[i].queueFlags & VK_QUEUE_TRANSFER_BIT) &&
					(vk.queueFamily_main != i || props[i].queueCount > 1)) // must have TRANSFER_BIT
				{
					vk.queueFamily_transfer = i;
					return;
				}
			}
			printf("Couldn't find an independent transfer queue\n");
		}();
	}

	{ // create device
		const float priorities[] = { 0.f };
		const vkh::CreateQueues createQueues[] = {
			{.familyIndex = vk.queueFamily_main, .priorities = priorities},
			{.familyIndex = vk.queueFamily_transfer, .priorities = priorities}
		};
		const u32 numCreateQueues = vk.queueFamily_transfer >= 0 ? 2 : 1;

		//vkEnumerateDeviceExtensionProperties(vk.physicalDevice, )
		ConstStr extensions[] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME,  };
		vk.device = vkh::createDevice(vk.physicalDevice, { createQueues, numCreateQueues }, extensions);

		vkGetDeviceQueue(vk.device, vk.queueFamily_main, 0, &vk.queue_main);
		if(vk.queueFamily_transfer >= 0)
			vkGetDeviceQueue(vk.device, vk.queueFamily_transfer, 0, &vk.queue_transfer);
	}

	// create allocator
	const VmaAllocatorCreateInfo allocatorInfo = {
		.flags = 0,
		.physicalDevice = vk.physicalDevice,
		.device = vk.device,
		.instance = vk.instance,
		.vulkanApiVersion = vulkanVersion,
	};
	vkRes = vmaCreateAllocator(&allocatorInfo, &vk.allocator);
	vkh::assertRes(vkRes);

	// create swapchain
	vkh::create_swapChain(vk.swapchain, vk.physicalDevice, vk.device, vk.surface, 2, VK_PRESENT_MODE_FIFO_KHR);
	for (u32 i = 0; i < vk.swapchain.numImages; i++)
		vk.framebuffers[i] = VK_NULL_HANDLE; // when we are about to draw, we will create the framebuffer if NULL

	// create semaphores and fences for swapchain synchronization
	vkh::createSemaphores(vk.device, { vk.semaphore_drawFinished, vk.swapchain.numImages });
	vkh::createSemaphores(vk.device, { vk.semaphore_swapchainImageAcquired, vk.swapchain.numImages });
	vkh::createFences(vk.device, true, { vk.fence_queueWorkFinished, vk.swapchain.numImages });

	vk.renderPass = vkh::createRenderPass_simple(vk.device, vk.swapchain.format.format);

	{ // create descriptor set layouts
		const VkDescriptorSetLayoutBinding cameraDescSet_bindings[] = { {
			.binding = 0,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
		} };
		vk.descSetLayout_camera = vkh::createDescriptorSetLayout(vk.device, cameraDescSet_bindings);

		const VkDescriptorSetLayoutBinding objectDescSet_bindings[] = {
			{
				.binding = 0,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.descriptorCount = 1,
				.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
			},
			{
				.binding = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.descriptorCount = 1,
				.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
			},
		};
		vk.descSetLayout_object = vkh::createDescriptorSetLayout(vk.device, objectDescSet_bindings);
	}

	vk.delayedDestrMan_descSet.init(vk.swapchain.numImages);

	// create pipeline layout
	const VkDescriptorSetLayout descSetLayouts[] = { vk.descSetLayout_camera, vk.descSetLayout_object };
	vk.pipelineLayout = vkh::createPipelineLayout(vk.device, descSetLayouts, {});

	// create pipeline
	const VkVertexInputBindingDescription vertexInputBindings[] = { {
		.binding = 0,
		.stride = sizeof(Vert),
		.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
	} };
	const VkVertexInputAttributeDescription vertexInputAttribs[] = {
		{
			.location = 0,
			.binding = 0,
			.format = VK_FORMAT_R32G32_SFLOAT,
			.offset = offsetof(Vert, pos),
		},
		{
			.location = 1,
			.binding = 0,
			.format = VK_FORMAT_R32G32_SFLOAT,
			.offset = offsetof(Vert, texCoord),
		},
	};
	const VkPipelineColorBlendAttachmentState attachmentBlendInfos[] = { {
		.blendEnable = VK_FALSE,
		.colorWriteMask = vkh::ALL_COLOR_COMPONENTS,
	} };
	const VkDynamicState dynamicStates[] = {
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR,
	};
	const vkh::CreateGraphicsPipeline pipelineCreateInfo = {
		.shaderStages = {
			.vertex = vkh::loadShaderModule(vk.device, SHADERS_FOLDER "simple.vert.spirv"),
			.fragment = vkh::loadShaderModule(vk.device, SHADERS_FOLDER "simple.frag.spirv"),
		},
		.vertexInputBindings = vertexInputBindings,
		.vertexInputAttribs = vertexInputAttribs,
		.primitiveTopology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		.attachmentsBlendInfos = attachmentBlendInfos,
		.dynamicStates = dynamicStates,
		.pipelineLayout = vk.pipelineLayout,
		.renderPass = vk.renderPass,
		.subpass = 0,
	};
	vk.pipeline = vkh::createGraphicsPipeline(vk.device, pipelineCreateInfo);

	// create command pools
	vk.cmdPool_main = vkh::createCmdPool(vk.device, vk.queueFamily_main, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
	if(vk.queueFamily_transfer >= 0)
		vk.cmdPool_transfer = vkh::createCmdPool(vk.device, vk.queueFamily_transfer, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

	// create cmd buffers
	vkh::allocateCmdBuffers(vk.device, vk.cmdPool_main, { vk.cmdBuffers_draw, vk.swapchain.numImages });
	vkh::allocateCmdBuffers(vk.device, vk.cmdPool_transfer, { &vk.cmdBuffer_load, 1 });

	// create staging buffer
	{
		const size_t size = 1024 * 1024 * sizeof(glm::u8vec4);
		vkh::createStagingBuffer(vk.device, vk.allocator, size, vk.stagingBuffer.buffer, vk.stagingBuffer.alloc, &vk.stagingBuffer.allocInfo);
	}
	size_t stagingBuffer_offset = 0;

	vkh::beginCmdBuffer(vk.cmdBuffers_draw[0]); // we will use this cmd buffer for initialization commands

	// create black image
	{
		vkh::ImgInfo info = {
			.width = 1, .height = 1,
			.mipLevels = 1
		};
		vkh::createImage_texture(vk.device, vk.allocator, info, vk.blackImg.img, vk.blackImg.alloc, &vk.blackImg.allocInfo, &vk.blackImg.view);

		// fill the staging buffer with the black pixel
		const glm::u8vec4 blackPixel = { 0,0,0, 255 };
		memcpy(vk.stagingBuffer.allocInfo.pMappedData, &blackPixel, sizeof(blackPixel));
		vmaFlushAllocation(vk.allocator, vk.stagingBuffer.alloc, stagingBuffer_offset, sizeof(blackPixel));
		stagingBuffer_offset += sizeof(blackPixel);

		recordCmds_copyBufferToImage(vk.cmdBuffers_draw[0], vk.blackImg.img, vk.stagingBuffer.buffer, 1, 1);
		recordCmd_transitionImageLayoutToShaderReadOnlyOptimal(vk.cmdBuffers_draw[0], vk.blackImg.img);
	}

	auto initCopyToBuffer = [&](const Buffer& buffer, CSpan<u8> data)
	{
		// copy to a buffer, using the staging buffer if needed
		if (auto mappedData = buffer.allocInfo.pMappedData) {
			memcpy(mappedData, data.data(), data.size());
			vmaFlushAllocation(vk.allocator, buffer.alloc, 0, VK_WHOLE_SIZE);
		}
		else {
			auto stagingBuffer_ptr = (u8*)vk.stagingBuffer.allocInfo.pMappedData;
			memcpy(stagingBuffer_ptr + stagingBuffer_offset, data.data(), data.size());
			vmaFlushAllocation(vk.allocator, vk.stagingBuffer.alloc, stagingBuffer_offset, data.size());

			const VkBufferCopy region_verts = {
				.srcOffset = stagingBuffer_offset,
				.dstOffset = 0,
				.size = data.size(),
			};
			vkCmdCopyBuffer(vk.cmdBuffers_draw[0], vk.stagingBuffer.buffer, buffer.buffer, 1, &region_verts);

			stagingBuffer_offset += data.size();
		}
	};
	// create vertex and index buffers
	{
		auto& cmdBuffer = vk.cmdBuffers_draw[0];
		// create vertex buffer
		const Vert verts[] = {
			{{-1, -1}, {0, 0}},
			{{-1, +1}, {0, 1}},
			{{+1, +1}, {1, 1}},
			{{+1, -1}, {1, 0}},
		};
		auto& vertBuffer = vk.quad.vertBuffer;
		vkh::createVertexBuffer(vk.device, vk.allocator, sizeof(verts), vertBuffer.buffer, vertBuffer.alloc, &vertBuffer.allocInfo);
		initCopyToBuffer(vertBuffer, {(u8*)verts, sizeof(verts)});

		// create index buffer
		const u32 inds[] = {
			0, 1, 3,
			1, 2, 3
		};
		auto& indBuffer = vk.quad.indBuffer;
		vkh::createIndexBuffer(vk.device, vk.allocator, sizeof(inds), indBuffer.buffer, indBuffer.alloc, &indBuffer.allocInfo);
		initCopyToBuffer(indBuffer, { (u8*)inds, sizeof(inds) });
	}

	// create uniforms buffers
	{
		const CameraParams cameraParams = { {0, 0}, {1, 0}, {0, 1} };
		vkh::createUniformBuffer(vk.device, vk.allocator, sizeof(cameraParams),
			vk.uniformBuffer_cameraParams.buffer, vk.uniformBuffer_cameraParams.alloc, &vk.uniformBuffer_cameraParams.allocInfo);
		initCopyToBuffer(vk.uniformBuffer_cameraParams, { (u8*)&cameraParams, sizeof(cameraParams) });

		const ObjectParams objectParams = { {0, 0}, {0.8, 0}, {0, 0.8} };
		vkh::createUniformBuffer(vk.device, vk.allocator, sizeof(objectParams),
			vk.uniformBuffer_objectParams.buffer, vk.uniformBuffer_objectParams.alloc, &vk.uniformBuffer_objectParams.allocInfo);
		initCopyToBuffer(vk.uniformBuffer_objectParams, { (u8*)&objectParams, sizeof(objectParams) });
	}

	{ // create descriptor pool
		const VkDescriptorPoolSize sizes[] = {
			{
				.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.descriptorCount = 2048,
			},
			{
				.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.descriptorCount = 1024,
			},
		};
		vk.descPool = vkh::createDescriptorPool(vk.device, 1024, sizes);
	}

	// create sampler
	{
		const VkSamplerCreateInfo info = {
			.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
			.magFilter = VK_FILTER_LINEAR,
			.minFilter = VK_FILTER_LINEAR,
			.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
			.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
			.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		};
		vkRes = vkCreateSampler(vk.device, &info, nullptr, &vk.sampler);
		vkh::assertRes(vkRes);
	}

	// create descriptor sets
	{
		vkRes = vkh::allocDescSets(vk.device, vk.descPool, { &vk.descSetLayout_camera, 1 }, { &vk.descSet_cameraParams, 1 });
		vkh::assertRes(vkRes);
		const vkh::DescSetWriteBuffer writeBuffers[] = {
			{
				.descSet = vk.descSet_cameraParams,
				.binding = 0,
				.buffer = vk.uniformBuffer_cameraParams.buffer,
			},
		};
		vkh::writeDescriptorSets(vk.device, writeBuffers, {});
	}
	{
		vkRes = vkh::allocDescSets(vk.device, vk.descPool, { &vk.descSetLayout_object, 1 }, { &vk.descSet_objectParams, 1 });
		vkh::assertRes(vkRes);
		const vkh::DescSetWriteBuffer writeBuffers[] = {
			{
				.descSet = vk.descSet_objectParams,
				.binding = 0,
				.buffer = vk.uniformBuffer_objectParams.buffer,
			},
		};
		const vkh::DescSetWriteTexture writeTextures[] = {
			{
				.descSet = vk.descSet_objectParams,
				.binding = 1,
				.imgView = vk.blackImg.view,
				.sampler = vk.sampler,
			}
		};
		vkh::writeDescriptorSets(vk.device, writeBuffers, writeTextures);
	}

	vkEndCommandBuffer(vk.cmdBuffers_draw[0]);

	// wait for completion of initialization commands because we need to reuse the cmdBuffer for drawing
	const VkFence fence = vkh::createFence(vk.device, false);
	vkh::submit(vk.queue_main, { &vk.cmdBuffers_draw[0], 1}, {}, {}, {}, fence);
	vkRes = vkWaitForFences(vk.device, 1, &fence, VK_FALSE, -1);
	vkh::assertRes(vkRes);

	Image tentImg = {};

	vk.loaderThread = std::thread(loaderThreadFn);

	loadImageAsync("data/tent.jpg", +[](void* userPtr, CStr fileName, const Image& img) {
		auto& tentImg = *(Image*)userPtr;
		tentImg = img;

		vk.delayedDestrMan_descSet.destroy(vk.descSet_objectParams); // queue for destruction

		VkResult vkRes = vkh::allocDescSets(vk.device, vk.descPool, { &vk.descSetLayout_object, 1 }, { &vk.descSet_objectParams, 1 });
		vkh::assertRes(vkRes);

		const vkh::DescSetWriteBuffer bufferDescWrites[] = { {
			.descSet = vk.descSet_objectParams,
			.binding = 0,
			.buffer = vk.uniformBuffer_objectParams.buffer,
		} };
		const vkh::DescSetWriteTexture textureDescWrites[] = { {
			.descSet = vk.descSet_objectParams,
			.binding = 1,
			.imgView = tentImg.view,
			.sampler = vk.sampler,
		} };
		vkh::writeDescriptorSets(vk.device, bufferDescWrites, textureDescWrites);
	}, &tentImg);

	u32 frameInd = 0;
	while (!glfwWindowShouldClose(vk.window))
	{
		glfwPollEvents();
		int screenW, screenH;
		glfwGetFramebufferSize(vk.window, &screenW, &screenH);
		if (screenW == 0 || screenH == 0)
			continue;

		u32 swapchainImageInd;
		vkRes = vkAcquireNextImageKHR(vk.device, vk.swapchain.swapchain, -1,
			vk.semaphore_swapchainImageAcquired[frameInd], VK_NULL_HANDLE, &swapchainImageInd);
		vkh::assertRes(vkRes);

		if (vk.framebuffers[swapchainImageInd] == VK_NULL_HANDLE) {
			vk.framebuffers[swapchainImageInd] = vkh::createFramebuffer(vk.device, vk.renderPass, { &vk.swapchain.imageViews[swapchainImageInd], 1 }, screenW, screenH);
		}

		vk.delayedDestrMan_descSet.startFrame([](const VkDescriptorSet& descSet) {
			vkFreeDescriptorSets(vk.device, vk.descPool, 1, &descSet);
		});

		vkWaitForFences(vk.device, 1, &vk.fence_queueWorkFinished[swapchainImageInd], VK_FALSE, -1);
		vkResetFences(vk.device, 1, &vk.fence_queueWorkFinished[swapchainImageInd]);
		auto& drawCmdBuffer = vk.cmdBuffers_draw[swapchainImageInd];
		vkh::beginCmdBuffer(drawCmdBuffer, true);
		{
			recordCmds_doPendingQueueTransferTasks(drawCmdBuffer);

			const VkClearValue clearVals[] = {{
				.color = {0, 0.05, 0, 1},
			} };
			vkh::cmdBeginRenderPass(drawCmdBuffer, vk.renderPass, vk.framebuffers[swapchainImageInd], screenW, screenH, clearVals);

			vkCmdBindPipeline(drawCmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vk.pipeline);

			const VkViewport viewport = {
				.x = 0, .y = 0,
				.width = float(screenW), .height = float(screenH),
				.minDepth = 0, .maxDepth = 1,
			};
			vkCmdSetViewport(drawCmdBuffer, 0, 1, &viewport);

			const VkRect2D scissor = { {0, 0}, {screenW, screenH} };
			vkCmdSetScissor(drawCmdBuffer, 0, 1, &scissor);

			size_t zeroOffset = 0;
			vkCmdBindVertexBuffers(drawCmdBuffer, 0, 1, &vk.quad.vertBuffer.buffer, &zeroOffset);
			vkCmdBindIndexBuffer(drawCmdBuffer, vk.quad.indBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
			const VkDescriptorSet descSets[] = { vk.descSet_cameraParams, vk.descSet_objectParams };
			vkCmdBindDescriptorSets(drawCmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vk.pipelineLayout, 0, std::size(descSets), descSets, 0, nullptr);
			vkCmdDrawIndexed(drawCmdBuffer, 6, 1, 0, 0, 0);

			vkCmdEndRenderPass(drawCmdBuffer);
		}
		vkEndCommandBuffer(drawCmdBuffer);
		const VkPipelineStageFlags colorOutputStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		vkh::submit(vk.queue_main, { &drawCmdBuffer, 1 },
			{ &vk.semaphore_swapchainImageAcquired[frameInd], 1 }, { &colorOutputStage, 1 }, // wait semaphores
			{ &vk.semaphore_drawFinished[swapchainImageInd], 1}, // signal semaphores
			vk.fence_queueWorkFinished[swapchainImageInd]);

		vkh::swapchainPresent(vk.queue_main, vk.swapchain.swapchain, swapchainImageInd, vk.semaphore_drawFinished[swapchainImageInd]);

		frameInd = (frameInd + 1) % vk.swapchain.numImages;
	}

	{
		std::lock_guard l(vk.loadTasks_mutex);
		vk.loadTasks_condVar.notify_one();
		vk.terminateLoaderThread = true;
	}
	vk.loaderThread.join();
	
}