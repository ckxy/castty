import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor


class GradCam(object):
	def __init__(self, model, point):
		self.features = None
		self.gradients = None
		self.handlers = []
		for name, module in model.named_modules():
			if name == point:
				self.handlers.append(module.register_forward_hook(self.forward_hook))
				self.handlers.append(module.register_backward_hook(self.backward_hook))
				break

	def remove(self):
		for handle in self.handlers:
			handle.remove()

	def forward_hook(self, module, fea_input, fea_output):
		self.features = fea_output

	def backward_hook(self, module, grad_input, grad_output):
		self.gradients = grad_output[0]

	def __call__(self, height, width):
		if self.features is not None and self.gradients is not None:
			gradient = self.gradients[0].detach()
			weight = torch.mean(gradient, (1, 2))

			feature = self.features[0].detach()

			cam = feature * weight.unsqueeze(-1).unsqueeze(-1)
			cam = torch.sum(cam, 0)

			cam = F.relu(cam)
			cam = cam - torch.min(cam)
			cam = cam / (torch.max(cam) + 1e-6)
			cam = cam.unsqueeze(0).unsqueeze(0)
			cam = F.interpolate(cam, size=(height, width), mode="bilinear")
			cam = cam.squeeze()
			cam = cam.cpu().numpy()

			heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
			heatmap = heatmap[:, :, ::-1].copy()

			heatmap = to_tensor(heatmap)

			return heatmap
		else:
			return torch.zeros(height, width, 3)


class GradCamPP(GradCam):
	def __init__(self, model, point):
		super(GradCamPP, self).__init__(model, point)

	def __call__(self, height, width):
		if self.features is not None and self.gradients is not None:
			gradient = self.gradients[0].detach()
			gradient = F.relu(gradient)
			indicate = torch.where(gradient > 0, torch.tensor(1.0).to(gradient.device), torch.tensor(0.0).to(gradient.device))
			norm_factor = torch.sum(gradient, (1, 2))

			for i in range(len(norm_factor)):
				norm_factor[i] = 1. / norm_factor[i] if norm_factor[i] > 0. else 0.
			alpha = indicate * norm_factor.unsqueeze(-1).unsqueeze(-1)

			weight = torch.sum(gradient * alpha, (1, 2))

			feature = self.features[0].detach()

			cam = feature * weight.unsqueeze(-1).unsqueeze(-1)
			cam = torch.sum(cam, 0)  # [H,W]
			# cam = np.maximum(cam, 0)  # ReLU

			cam = cam - torch.min(cam)
			cam = cam / (torch.max(cam) + 1e-6)
			cam = cam.unsqueeze(0).unsqueeze(0)
			cam = F.interpolate(cam, size=(height, width), mode="bilinear")
			cam = cam.squeeze()
			cam = cam.cpu().numpy()

			heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
			heatmap = heatmap[:, :, ::-1].copy()

			heatmap = to_tensor(heatmap)

			return heatmap
		else:
			return torch.zeros(height, width, 3)


class GuidedBackPropagation(object):
	def __init__(self, models):
		self.handlers = []
		for model in models:
			for name, module in model.named_modules():
				if isinstance(module, nn.ReLU) or isinstance(module, nn.ReLU6) or isinstance(module, nn.LeakyReLU):
					self.handlers.append(module.register_backward_hook(self.backward_hook))

	def remove(self):
		for handle in self.handlers:
			handle.remove()

	def backward_hook(self, module, grad_in, grad_out):
		return torch.clamp(grad_in[0], min=0.0),

	def __call__(self, x):
		grad = x.grad[0]
		grad = grad.detach()
		grad = grad - torch.min(grad)
		grad = grad / (torch.max(grad) + 1e-6)
		# grad = (grad * 255).astype(np.uint8)
		return grad


class IntegratedGradients(object):
	def __init__(self, model, steps=100):
		self.steps = steps
		self.steps_list = np.arange(steps + 1) / steps
		self.gradients = None

		for name, module in model.named_modules():
			if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
				self.handler = module.register_backward_hook(self.backward_hook)
				break

	def remove(self):
		self.handler.remove()

	def backward_hook(self, module, grad_input, grad_output):
		if self.gradients is None:
			self.gradients = grad_input[0] / self.steps
		else:
			self.gradients += grad_input[0] / self.steps

	def __call__(self):
		if self.gradients is not None:
			grad = self.gradients[0]
			grad = grad.detach()
			grad = grad - torch.min(grad)
			grad = grad / (torch.max(grad) + 1e-6)
			self.gradients = None

			return grad.cpu()
		else:
			return None
