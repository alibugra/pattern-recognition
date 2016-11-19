classdef ExpectationMax
    properties
        k = 2 % Change this K = 2, K = 3 etc.
        i = 1
        test_image_path = 'img/test_image.png'
    end
    methods
        function r = run(obj)
            img = imread(obj.test_image_path);
            img_vector = img(:);
            data = double(transpose(img_vector));

            [mu, sigma] = obj.em_step(data, obj.k);
            fx = obj.gaussian_distribution(data, mu, sigma);
            max_value = max(fx);
            dl = max_value - fx(1, :);
            dl(dl ~= 0) = 2;
            dl(dl == 0) = 1;

            disp(dl);
            disp(mu);
            disp(sigma);
        end
        
        function [mu, sigma] = em_step(obj, data, k)
            temp = randperm(length(data));
            weight(1:k, 1) = (1 / k) * ones(k, 1);
            mu(1:k, 1) = data(temp(1:k));
            sigma(1:k, 1) = var(data);

            % E Step
            ll_temp = 0;
            while(obj.i < 250)
                p = zeros(1, length(data));
                probability = obj.gaussian_distribution(data, mu, sigma);
                p = (weight'*probability);
                log_likelihood(obj.i) = sum(log(p));

                % M Step
                mult = bsxfun(@times, weight, probability);
                post = bsxfun(@rdivide, mult, p);
                mu = (post * data')./((sum(post'))');
                dmm = bsxfun(@minus, data, mu).^2;
                sigma = (sum((post.*(dmm))')./sum(post'))';
                weight = (sum(post')./length(data))';
                logLikelihoodThreshold = abs(ll_temp - log_likelihood(obj.i)) < 10^-10;
                if logLikelihoodThreshold
                    break;
                end
                obj.i = obj.i + 1;
            end
            
            obj.plot_likelihood(log_likelihood);
        end
        
        function [fx] = gaussian_distribution(obj, x, mu, sigma)
            xmmu = bsxfun(@minus, x, mu);
            division = bsxfun(@rdivide, ((xmmu).^2), (2.*sigma));
            ex = exp(-(division));
            fx = bsxfun(@times, (1./(sqrt(2*pi.*sigma))), ex);
        end
        
        function pl = plot_likelihood(obj, log_likelihood)
            plot(log_likelihood)
            xlabel('Iteration');
            ylabel('Data Log-likelihood');
        end
    end
end