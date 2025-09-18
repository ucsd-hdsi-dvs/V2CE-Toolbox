import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur
from torchvision.transforms import Grayscale
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d

def get_image_gradient(image1):
    """
    Calculate image gradient using Sobel filter.
    Args:
        image1: np.ndarray, shape (h, w), the first image.
        image2: np.ndarray, shape (h, w), the second image.
    Returns:
        gradient: np.ndarray, shape (h, w), the image gradient.
    """
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = sobel_x.T
    grad_x = convolve2d(image1, sobel_x, mode='same')
    grad_y = convolve2d(image1, sobel_y, mode='same')
    gradient = np.sqrt(grad_x ** 2 + grad_y ** 2)
    return gradient

def get_double_blurred_image_gradient(image1, image2, sigma=3):
    # Calculate image gradient using Sobel filter.
    image1_gradient = get_image_gradient(image1)
    image2_gradient = get_image_gradient(image2)

    # merge using point-wise maximum
    img_grad_merge = np.maximum(image1_gradient, image2_gradient)

    # Expand the image gradient coverage area by applying Gaussian filter.
    img_grad_merge_blur = gaussian_filter(img_grad_merge, sigma=sigma)
    return img_grad_merge_blur

def get_batch_double_blurred_image_gradient(image1, image2, sigma=3, kernel_size=11):
    """
    Calculate image gradient using Sobel filter.
    Args:
        image1: np.ndarray, shape (b, c, h, w), the first image.
        image2: np.ndarray, shape (b, c, h, w), the second image.
    Returns:
        gradient: np.ndarray, shape (b, h, w), the image gradient.
    """
    # Calculate image gradient using Sobel filter.
    image1_gradient = batch_img_gradient(image1)
    image2_gradient = batch_img_gradient(image2)

    # merge using point-wise maximum
    img_grad_merge = torch.maximum(image1_gradient, image2_gradient)

    # Expand the image gradient coverage area by applying Gaussian filter, using torch
    img_grad_merge_blur = gaussian_blur(img_grad_merge, kernel_size=kernel_size, sigma=sigma)
    return img_grad_merge_blur

def batch_img_gradient(img):
    """
    Calculate the image gradient using Sobel filter in batch using pytorch.
    Args:
        img: torch.Tensor, shape (b, c, h, w), the input image.
    Returns:
        grad: torch.Tensor, shape (b, c, h, w), the image gradient.
    """
    b, c, h, w = img.shape

    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).repeat(c, 1, 1, 1)
    sobel_y = sobel_x.permute(0, 1, 3, 2)
    sobel_x = sobel_x.to(img.device)
    sobel_y = sobel_y.to(img.device)
    grad_x = F.conv2d(img, sobel_x, padding=1)
    grad_y = F.conv2d(img, sobel_y, padding=1)
    grad = torch.sqrt(grad_x ** 2 + grad_y ** 2)
    return grad

def batch_img_residual(img1, img2):
    """
    Calculate the image residual between two images.
    Args:
        img1: torch.Tensor, shape (b, c, h, w), the first image.
        img2: torch.Tensor, shape (b, c, h, w), the second image.
    Returns:
        residual: torch.Tensor, shape (b, c, h, w), the image residual.
    """
    residual = img2 - img1
    return residual

def batch_image_derivative_calc(img1, img2, ofc):
    """
    Calculate the image derivative in batch.
    Args:
        img1: torch.Tensor, shape (b, c, h, w), the first image.
        img2: torch.Tensor, shape (b, c, h, w), the second image.
        ofc: OpticalFlowCalculator, the optical flow calculator.
    Returns:
        image_derivative: torch.Tensor, shape (b, 5, h, w), the image derivative.
    """
    rgb2gray = Grayscale()
    img1_gray = rgb2gray(img1)
    img2_gray = rgb2gray(img2)

    img_residual = batch_img_residual(img1_gray, img2_gray)
    img1_gradient = batch_img_gradient(img1_gray)
    img2_gradient = batch_img_gradient(img2_gray)
    img_flow = ofc(img1, img2).cpu()
    image_derivative = torch.cat([img_residual, img1_gradient, img2_gradient, img_flow], dim=1)
    return image_derivative

def single_image_derivative_calc(img1_path, img2_path, ofc):
    """
    Calculate the image derivative for a pair of images.
    Args:
        img1_path: str, the path of the first image.
        img2_path: str, the path of the second image.
        ofc: OpticalFlowCalculator, the optical flow calculator.
    Returns:
        image_derivative: torch.Tensor, shape (1, 5, h, w), the image derivative.
    """
    img1 = cv2.imread(img1_path).astype(np.float32) / 255.
    img2 = cv2.imread(img2_path).astype(np.float32) / 255.

    img1 = torch.from_numpy(img1).float().permute(2, 0, 1).unsqueeze(0).cuda()
    img2 = torch.from_numpy(img2).float().permute(2, 0, 1).unsqueeze(0).cuda()
    image_derivative = batch_image_derivative_calc(img1, img2, ofc)
    return image_derivative
