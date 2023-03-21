classdef PPISCS
    methods (Static)
        function y = swish(x)
            y = x./(1 + exp(-x));
        end
        
        function y = softplus(x)
            y = log(1 + exp(-abs(x))) + max(x, 0);
        end
        
        function y = eval_on_batch(y, layer_bias, layer_weights)
            n_layers = length(layer_bias);
            y = PPISCS.swish(y*layer_weights{1} + layer_bias{1});
            for i = 2:n_layers-1
                y = [y, PPISCS.swish(y*layer_weights{i} + layer_bias{i})];
            end
            y = y*layer_weights{end} + layer_bias{end};
            y = PPISCS.softplus(y);
        end
        
        function y = eval(y_i, bias, weights, batch_size)
            n_y = size(y_i, 1);
            if n_y <= batch_size
                y = PPISCS.eval_on_batch(y_i, bias, weights);
                return
            end
            
            y = zeros(n_y, numel(bias{end}));
            for ik = 1:batch_size:n_y
                y(ik:ik+batch_size-1, :) = PPISCS.eval_on_batch(y_i(ik:ik+batch_size-1, :), bias, weights);
            end
        end
    end
    
    methods
        function obj = PPISCS(model_path)
            if nargin < 1
                model_path = 'coef_scs_fcc.mat';
            end
            model = importdata(model_path, 'model');
            obj.bias = model.bias;
            obj.weights = model.weights;
            obj.x_sft = model.x_sft;
            obj.x_sc = model.x_sc;
        end
        
        function y = predict(obj, input_data, batch_size, crop)
            if nargin < 4
                crop = true;
            end
            if nargin < 3
                batch_size = 4096*4;
            end
            if ~ismatrix(input_data) || (ndims(input_data) ~= 2) || (size(input_data, 2) ~= 9) %#ok<ISMAT> 
                error('Input data must be a 2D double array with 9 columns')
            end
            
            if crop && input_data(1, 2) == 1
                crop_end = 61;
            else
                crop_end = [];
            end
            
            y = (double(input_data) - obj.x_sft)./obj.x_sc;
            y = obj.eval(y, obj.bias, obj.weights, batch_size);
            if ~isempty(crop_end)
                y = y(:, 1:crop_end);
            end
        end
    end
    
    properties (Access = private)
        bias
        weights
        x_sft
        x_sc
    end
end
