import tensorflow as tf


class Preprocessor(object):
    def __init__(self):
        pass

    @staticmethod
    def __augmentation_random_crop(img, crop_size):
        """
        random crop image
        :param img: image_tensor
        :param crop_size: crop_size_list
        :return: augmented_image_tensor
        """
        if type(crop_size) != list:
            raise TypeError('crop_size must be list [crop_height, crop_width, channels]')
        if crop_size[0] < 0 or crop_size[1] < 0:
            raise ValueError('Crop size must be bigger than 0 and smaller than the origin image size')
        # Since the tf.random_crop op doesn't support 4-D Tensor with batch, so use tf.map_fn() to operate on each
        # element
        result = tf.map_fn(lambda image: tf.random_crop(value=image, size=crop_size), img)
        return result

    @staticmethod
    def __augmentation_random_flip_horizon(img):
        """
        flip image horizonlly
        :param img: image_tensor
        :return: augmented_image_tensor
        """
        result = tf.map_fn(lambda image: tf.image.random_flip_left_right(image=image), img)
        return result

    @staticmethod
    def __augmentation_random_filp_vertical(img):
        """
        flip image vertically
        :param img: image tensor
        :return: augmented_image_tensor
        """
        result = tf.map_fn(lambda image: tf.image.random_flip_up_down(image=image), img)
        return result

    @staticmethod
    def __augmentation_random_brightness(img, brightness):
        """
        random add brightness noise to image and the brightness varies from [-brightess, brightness)
        :param img: origin image tensor
        :param brightness: brightness noise to be added brightness varies from [-brightess, brightness)
        :return: augmented_image_tensor
        """
        result = tf.map_fn(lambda image: tf.image.random_brightness(image=image, max_delta=brightness), img)
        return result

    @staticmethod
    def __augmentation_random_contrast(img, lower_factor, upper_factor):
        """
        randomly change the contrast of the image, change factor constrast_factor varies from [lower_factor, upper_factor]
        For each channel, this Op computes the mean of the image pixels in the channel and then adjusts each component
        x of each pixel to (x - mean) * contrast_factor + mean
        :param image: image tensor
        :param lower_factor: lowest constrast factor
        :param upper_factor: uppest constrast factor
        :return: augmented_image_tensor
        """
        result = tf.map_fn(lambda image: tf.image.random_contrast(image=image, lower=lower_factor, upper=upper_factor),
                           img)
        return result

    @staticmethod
    def __augmentation_std_normalization(img):
        """
        Subtract off the mean and divide by the variance of the pixels.(std normalization)
        :param image: origin image tensor
        :return: augmented_image_tensor
        """
        result = tf.map_fn(lambda image: tf.image.per_image_standardization(image=image), img)
        return result

    @staticmethod
    def __augmentation_minmax_normalization(img):
        """
        op: use (pixel - min) / (max - min) to do the normalization
        :param img: image_tensor
        :return: augmented_image_tensor
        """

        def __min_max_norm(image_single):
            pixel_max_tensor = tf.reduce_max(input_tensor=image_single, reduction_indices=[0, 1])
            pixel_min_tensor = tf.reduce_min(input_tensor=image_single, reduction_indices=[0, 1])
            image_single = tf.divide(tf.subtract(image_single, pixel_min_tensor),
                                     tf.subtract(pixel_max_tensor, pixel_min_tensor))
            return image_single

        result = tf.map_fn(lambda image: __min_max_norm(image_single=image), img)
        return result

    @staticmethod
    def __augmentation_centralization(img, center_value):
        """
        Image whiten process new_value = origin_value - center_value
        :param img: origin image
        :param center_value: value used to centeralization eg. for ImageNet [104, 117, 123]
        :return:
        """

        def __centralization(image_single):
            mean_value = tf.constant(value=center_value, dtype=tf.float32, shape=[3], name='Image_Mean_Value')
            return tf.subtract(image_single, mean_value)

        result = tf.map_fn(lambda image: __centralization(image_single=image), img)
        return result

    @staticmethod
    def augment_image(self, image, augment_para_dict):
        """
        parase augment_para_dict to do data augmentation
        :param self: class itself
        :param image: origin_image_tensor
        :param augment_para_dict: for example
        augment_dict = {'flip_horizon': True,
                        'flip_vertical': True,
                        'random_crop': {
                        'need_random_crop': True
                        'crop_size': [227, 227, 3]
                        }
                        }
        :return: augmented_image_tensor
        """
        if augment_para_dict['flip_horizon']:
            image = self.__augmentation_random_flip_horizon(img=image)

        if augment_para_dict['flip_vertical']:
            image = self.__augmentation_random_filp_vertical(img=image)

        if augment_para_dict['random_crop']['need_random_crop']:
            crop_size = augment_para_dict['random_crop']['crop_size']
            image = self.__augmentation_random_crop(img=image, crop_size=crop_size)

        if augment_para_dict['random_brightness']['need_random_brightness']:
            brightness = augment_para_dict['random_brightness']['brightness']
            image = self.__augmentation_random_brightness(img=image, brightness=brightness)

        if augment_para_dict['random_contrast']['need_random_contrast']:
            lower_factor = augment_para_dict['random_contrast']['lower_factor']
            upper_factor = augment_para_dict['random_contrast']['upper_factor']
            image = self.__augmentation_random_contrast(
                self=self, img=image, lower_factor=lower_factor,
                upper_factor=upper_factor)

        if augment_para_dict['centralization']['need_centralization']:
            center_value = augment_para_dict['centralization']['mean_value']
            image = self.__augmentation_centralization(img=image, center_value=center_value)

        if augment_para_dict['std_normalization']:
            image = self.__augmentation_std_normalization(img=image)

        if augment_para_dict['minmax_normalization']:
            image = self.__augmentation_minmax_normalization(img=image)
        return image
