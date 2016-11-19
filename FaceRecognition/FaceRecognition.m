classdef FaceRecognition
    properties
        images = []
        image_dimensions = [50, 50]
        training_set = 'C:\Users\alibugra\Desktop\FaceRecognition\training'
        test_image = 'C:\Users\alibugra\Desktop\FaceRecognition\test\1.png'
        num_eigenfaces = 12
    end
    methods
        function r = run(obj)
            image_files = dir(fullfile(obj.training_set, '*.png'));
            number_img_elems = numel(image_files);
            obj.images = zeros(prod(obj.image_dimensions), number_img_elems);
            
            for i = 1:number_img_elems
                filename = fullfile(obj.training_set, image_files(i).name);
                img = imread(filename);
                img = rgb2gray(img);
                img = im2double(img);
                img = imresize(img, obj.image_dimensions);
                obj.images(:, i) = img(:);
            end
            
            mean_face = mean(obj.images, 2);
            fi_img = obj.images - repmat(mean_face, 1, number_img_elems);
            [eigenvecs, score, eigenvalues] = princomp(obj.images', 'econ');
            eigenvecs = eigenvecs(:, 1:obj.num_eigenfaces);
            wi = eigenvecs' * fi_img;

            test_set_image = imread(obj.test_image);
            test_set_image = rgb2gray(test_set_image);
            test_set_image = imresize(test_set_image, obj.image_dimensions);
            test_set_image = im2double(test_set_image);
            
            feature_vec = eigenvecs' * (test_set_image(:) - mean_face);
            recognized_distance = arrayfun(@(i) 1 / (1 + norm(wi(:, i) - feature_vec)), 1:number_img_elems);
            [rec_score, recognized_file] = max(recognized_distance);

            figure, imshow([test_set_image reshape(obj.images(:, recognized_file), obj.image_dimensions)]);
            title(sprintf('Recognized %s', image_files(recognized_file).name));
            figure;
            for i = 1:obj.num_eigenfaces
                subplot(2, ceil(obj.num_eigenfaces / 2), i);
                eigenvector = reshape(eigenvecs(:, i), obj.image_dimensions);
                imagesc(eigenvector);
                colormap(gray);
            end
        end
    end
end