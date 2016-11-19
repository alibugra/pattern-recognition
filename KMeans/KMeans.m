classdef KMeans
    properties
        j = 1
        num_of_clust = 3 % Change this K = 2, K = 3 etc.
        max_iteration = 10
        test_image_path = 'img/test_image.png'
    end
    methods
        function r = run(obj)
            while (obj.j <= obj.max_iteration)
                test_image = double(imread(obj.test_image_path)) / 255;
                image_size = size(test_image);
                reshaped_image = reshape(test_image, image_size(1) * image_size(2), 3);
                
                ct = 0;
                centroids = obj.initCentroids(reshaped_image, obj.num_of_clust);
                
                for num = 1:obj.max_iteration
                    C = obj.euclidean(reshaped_image, centroids);
                    centroids = obj.updateCentroids(reshaped_image, C, obj.num_of_clust);
                end
                
                for i = 1:obj.num_of_clust
                    tmp = find(C == i);
                    tmp2 = reshaped_image(tmp, :);
                    diff = bsxfun(@minus, centroids(i, :), tmp2);
                    ct = ct + (1 / length(tmp2)) * sum((sum(((diff).^2),2)).^(-0.5));
                end
                
                if (obj.j == 1) || (tmpC > ct && obj.j > 1)
                    bestCentroids = centroids;
                    tmpC = ct;
                    bestC = C;
                end
                
                compressed_image = centroids(C, :);
                compressed_image = reshape(compressed_image, image_size(1), image_size(2), 3);
                imshow(compressed_image);
                fprintf('Iteration %d \n', obj.j);
                pause(1);
                obj.j = obj.j + 1;
            end
            
            compressed_image = bestCentroids(bestC, :);
            compressed_image = reshape(compressed_image, image_size(1), image_size(2), 3);
            
            subplot(1, 2, 1);
            imshow(test_image);
            title('Image');
            subplot(1, 2, 2);
            imshow(compressed_image);
            title(sprintf('K = %d', obj.num_of_clust));
        end
        
        function initialize_centroids = initCentroids(obj, X, K)
            tmp = (randperm(length(X)))';
            initialize_centroids = X(tmp(1:K, 1), :);
        end
        
        function centroids = updateCentroids(obj, X, C, K)
            [m n] = size(X);
            centroids = zeros(K, n);
            for i = 1:K
                tmp = find(C == i);
                tmp2 = X(tmp, :);
                centroids(i, :) = (sum(tmp2, 1))./length(tmp2);
            end
        end
        
        function e = euclidean(obj, X, centroids)
            K = size(centroids, 1);
            e = zeros(size(X, 1), 1);
            for i = 1 : size(X, 1)
                tmp = X(i, :);
                diff = bsxfun(@minus, tmp, centroids);
                [~, e(i, 1)] = min(sum(((diff).^2), 2));
            end
        end
    end
end