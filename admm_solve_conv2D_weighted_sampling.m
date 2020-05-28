function [ z, res, res_b] = admm_solve_conv2D_weighted_sampling(b, video_size, kernels , mask, psf, ...
                    lambda_residual, lambda_prior, lambda_temp, gamma_temp, smooth_init, ...
                    max_it, tol, ...
                    x_orig, verbose)
                
    %Kernel matrix
    kmat = kernels;  
    
    %Precompute spectra for H (assumed to be convolutional)
    psf_radius = floor( [size(kmat,1)/2, size(kmat,2)/2] );
    size_x = [size(b,1) + 2*psf_radius(1), size(b,2) + (video_size(3)+1)*psf_radius(2)];
    [dhat_k, dhat, dhat_flat, dhatTdhat_flat] = precompute_H_hat(kmat, psf, size_x);
    dhatT_flat = conj(dhat_flat.');
    
    % Temporal forward differences
    Mf = video_size(1);
    Nf = video_size(2);
    T = video_size(3);
%     aux = Mf*Nf + psf_radius(1)*(2*Nf + Mf + 2*psf_radius(1)) ;
%     delta = sparse(aux*T + (2*psf_radius(1)+Mf)*psf_radius(1), aux*T + (2*psf_radius(1)+Mf)*psf_radius(1));
%     a_eye = speye(aux*(T-1), aux*(T-1))*-1;
%     b_eye = speye(aux*(T-1), aux*(T-1));
%     A = sparse(aux*T + (2*psf_radius(1)+Mf)*psf_radius(1), aux*T + (2*psf_radius(1)+Mf)*psf_radius(1));
%     %%%%FIX!!!! A = B y una tiene que estar desplazada
%     B = sparse(aux*T + (2*psf_radius(1)+Mf)*psf_radius(1), aux*T + (2*psf_radius(1)+Mf)*psf_radius(1));
%     
%     A((aux+1):(end - (2*psf_radius(1)+Mf)*psf_radius(1)) , 1 : (end- aux - (2*psf_radius(1)+Mf)*psf_radius(1))) = a_eye;
%     B((aux+1):(end - (2*psf_radius(1)+Mf)*psf_radius(1)) , aux+1 : (end-  (2*psf_radius(1)+Mf)*psf_radius(1))) = b_eye;
%     clear a_eye b_eye
%     delta(A==-1) = A(A==-1);
%     delta(B==1) = B(B==1);
%     clear A B
    
    %Size of z is now the padded array
    size_z = [size_x(1), size_x(2), size(kmat, 3)];
    
    % Objective
    objective = @(v) objectiveFunction( v, dhat, b, mask, lambda_residual, lambda_prior, psf_radius, video_size );
    
    %Proximal terms
    conv_term = @(xi_hat, gammas) solve_conv_term(dhat_flat, dhatT_flat, dhatTdhat_flat, xi_hat, gammas, size_z);
    
    %Smooth offset
    smoothinit = zeros(video_size(1) + psf_radius(1), video_size(3)*video_size(2) + psf_radius(1)*(video_size(3)));
    for i=1:video_size(3)
        smoothiniti = padarray(smooth_init(:,(i-1)*video_size(2) + 1 : i*video_size(2) ), psf_radius, 'symmetric', 'post');
        smoothinit(:, (i-1)*(video_size(2)+psf_radius(1)) + 1 : i*(video_size(2)+psf_radius(1))) = smoothiniti;     
    end
    smoothinit = padarray(smoothinit, psf_radius, 'symmetric', 'pre');
    
%     smoothinit = padarray( smooth_init, psf_radius, 'symmetric', 'both');
    
    %Prox for masked data
    [MtM, Mtb] = precompute_MProx(b, smoothinit, mask, psf_radius, video_size);
    ProxDataMasked = @(u, theta) (Mtb + 1/theta * u ) ./ ( MtM + 1/theta * ones(size_x) ); 
    
    %Prox for sparsity
    ProxSparse = @(u, theta) max( 0, 1 - theta./ abs(u) ) .* u;
    
    %Prox for temporal smoothness
     ProxTemp = @(u, theta) max( 0, 1 - theta./ abs(u) ) .* u;

    %alpha = 2/3;
    %ProxSparse = @(u, theta) solve_image(u, 1/theta, alpha);
    
    %Pack lambdas and find algorithm params
    lambda = [lambda_residual, lambda_prior, lambda_temp];
    gamma_heuristic = 20 * lambda_prior * 1/max(b(:));
    gamma = [gamma_heuristic / 5, gamma_heuristic, gamma_heuristic*gamma_temp];
    %gamma = [gamma_heuristic / 100 , gamma_heuristic];
    
    %Initialize variables
    varsize = {size_x, size_z};
    xi = { zeros(varsize{1}), zeros(varsize{2}), zeros(varsize{2}) };
    xi_hat = { zeros(varsize{1}), zeros(varsize{2}), zeros(varsize{2}) };
    
    u = { zeros(varsize{1}), zeros(varsize{2}), zeros(varsize{2})};
    d = { zeros(varsize{1}), zeros(varsize{2}), zeros(varsize{2})};
    v = { zeros(varsize{1}), zeros(varsize{2}), zeros(varsize{2})};
    
    %Initial iterate
    z = zeros(varsize{2});
    z_hat = zeros(varsize{2});
    
    %Debug
    if strcmp(verbose, 'brief') || strcmp( verbose, 'all')

        Dz = real(ifft2( sum( dhat_k .* z_hat, 3) ));
%         Dz = Dz(1 + psf_radius(1):end - psf_radius(1),1 + psf_radius(2):end - psf_radius(2),:);
% 
%         psnr_pad = psf_radius;
%         I_diff = x_orig(psnr_pad + 1:end - psnr_pad, psnr_pad + 1:end - psnr_pad,:) - Dz(psnr_pad + 1:end - psnr_pad, psnr_pad + 1:end - psnr_pad,:);
        Dz_disp = zeros(video_size(1), video_size(2)*video_size(3));
        for k=1:video_size(3)
            Dz_disp(:, (k-1)*video_size(2)+1:k*video_size(2)) = Dz(1 + psf_radius(1):end - psf_radius(1), (k-1)*(video_size(2))+ k*psf_radius(1)+ 1 : k*(video_size(2)+psf_radius(1)));
        end
        I_diff = x_orig - Dz_disp;
        MSE = 1/size(I_diff(:),1)*(norm(I_diff(:), 2)^2);
        if MSE > eps
            PSNR_ours = 10*log10(1/MSE);
        else
            PSNR_ours = Inf;
        end

        obj_val = objective(z);
        fprintf('Iter %d, Obj %3.3g, PSNR %2.2f, Diff %5.5g\n', 0, obj_val, PSNR_ours, 0)
    end
    
    %Display it.
    if strcmp(verbose, 'all')  
        iterate_fig = figure();
        Dz = real(ifft2( sum( dhat .* z_hat, 3) )) + smoothinit;
        Dz_disp = zeros(video_size(1), video_size(2)*video_size(3));
        for k=1:video_size(3)
            Dz_disp(:, (k-1)*video_size(2)+1:k*video_size(2)) = Dz(1 + psf_radius(1):end - psf_radius(1), (k-1)*(video_size(2))+ k*psf_radius(1)+ 1 : k*(video_size(2)+psf_radius(1)));
        end
%         Dz = Dz(1 + psf_radius(1):end - psf_radius(1),1 + psf_radius(2):end - psf_radius(2),:);
        
        subplot(1,2,1), imagesc(x_orig), axis image, colormap gray; title('Orig');
        subplot(1,2,2), imagesc(Dz_disp), axis image, colormap gray; title(sprintf('Local iterate %d',0));
    end

    
    %Iterate
    for i = 1:max_it
        
        %Compute v_i = H_i * z
        v{1} = real(ifft2( sum( dhat .* z_hat, 3)));
        v{2} = z;
        
        zdisp = zeros(video_size(1), video_size(2)*video_size(3), size_z(3));
        for k=1:video_size(3)
            zdisp(:, (k-1)*video_size(2)+1:k*video_size(2), :) = z(1 + psf_radius(1):end - psf_radius(1), (k-1)*(video_size(2))+ k*psf_radius(1)+ 1 : k*(video_size(2)+psf_radius(1)), :);
        end
        
        tempdifdisp = zeros(Mf, Nf, T, size_z(3));
        zdisp = reshape(zdisp, Mf, Nf, T, size_z(3));
        for k=1:size_z(3)
            aux = zdisp(:,:,:,k);
            for kk=2:size(aux,3)
                tempdifdisp(:,:,kk,k) = aux(:,:,kk)-aux(:,:,kk-1);
            end
%             tempdifdisp(:,:,:,k) = cat(3, zeros(size(aux(:,:,1))), tempdifdisp (:,:,1:T-1,k));
        end
        tempdifdisp = reshape(tempdifdisp,Mf, Nf*T, size_z(3));
%         figure;imagesc(tempdifdisp(:,:,1)); axis image
        tempdif = zeros(size_z(1), size_z(2), size_z(3));
        for nf=1:size_z(3)
            tempdiff = zeros(video_size(1) + psf_radius(1), video_size(3)*video_size(2) + psf_radius(1)*(video_size(3)));
            for k=1:video_size(3)
                tempdifi = tempdifdisp(:,:,nf);
                tempdifi = padarray(tempdifi(:,(k-1)*video_size(2) + 1 : k*video_size(2)), psf_radius, 'symmetric', 'post');
                tempdiff(:, (k-1)*(video_size(2)+psf_radius(1)) + 1 : k*(video_size(2)+psf_radius(1))) = tempdifi;     
            end
             tempdiff = padarray(tempdiff, psf_radius, 'symmetric', 'pre');
             tempdif(:,:,nf) = tempdiff;
        end
        
        v{3} = tempdif; 
        
        
        
%         aux = reshape(z, size_z(1)*size_z(2), size_z(3));
%         for k = 1:size_z(3)
%             tempdif(:,k) = delta*(aux(:,k));
%         end
%         v{3} = reshape(tempdif, size_z(1), size_z(2), size_z(3));
        
        %Compute proximal updates
        u{1} = ProxDataMasked( v{1} - d{1}, lambda(1)/gamma(1) );
        
        u{2} = ProxSparse( v{2} - d{2}, lambda(2)/gamma(2) );
        u{3} = ProxTemp(v{3} - d{3}, lambda(3) / gamma(3));
        
        for c = 1:3
            %Update running errors
            d{c} = d{c} - (v{c} - u{c});

            %Compute new xi and transform to fft
            xi{c} = u{c} + d{c};
            xi_hat{c} = fft2(xi{c});
        end

        %Solve convolutional inverse
        % z = ( sum_j(gamma_j * H_j'* H_j) )^(-1) * ( sum_j(gamma_j * H_j'* xi_j) )
        zold = z;
        z_hat = conv_term( xi_hat, gamma );
        z = real(ifft2( z_hat ));
        
        z_diff = z - zold;
        z_comp = z;
        
        %Debug
        if strcmp(verbose, 'brief') || strcmp( verbose, 'all')
            
            Dz = real(ifft2( sum( dhat_k .* z_hat, 3) )) + smoothinit;
%             Dz = Dz(1 + psf_radius(1):end - psf_radius(1),1 + psf_radius(2):end - psf_radius(2),:);
%             
%             psnr_pad = psf_radius;
%             I_diff = x_orig(psnr_pad + 1:end - psnr_pad, psnr_pad + 1:end - psnr_pad,:) - Dz(psnr_pad + 1:end - psnr_pad, psnr_pad + 1:end - psnr_pad,:);
            Dz_disp = zeros(video_size(1), video_size(2)*video_size(3));
            for k=1:video_size(3)
                Dz_disp(:, (k-1)*video_size(2)+1:k*video_size(2)) = Dz(1 + psf_radius(1):end - psf_radius(1), (k-1)*(video_size(2))+ k*psf_radius(1)+ 1 : k*(video_size(2)+psf_radius(1)));
            end
            I_diff = x_orig - Dz_disp;
            MSE = 1/size(I_diff(:),1)*(norm(I_diff(:), 2)^2);
            if MSE > eps
                PSNR_ours = 10*log10(1/MSE);
            else
                PSNR_ours = Inf;
            end

            obj_val = objective(z);
            fprintf('Iter %d, Obj %3.3g, PSNR %2.2f, Diff %5.5g\n', i, obj_val, PSNR_ours, norm(z_diff(:),2)/ norm(z_comp(:),2))
        end
        
        %Display it.
        if strcmp(verbose, 'all')
            
            figure(iterate_fig);
            subplot(1,2,1), imagesc(x_orig), axis image, colormap gray; title('Orig');
            subplot(1,2,2), imagesc(Dz_disp), axis image, colormap gray; title(sprintf('Local iterate %d',i));
        end
        
        if norm(z_diff(:),2)/ norm(z_comp(:),2) < tol
            break;
        end
    end
    
    Dz = real(ifft2( sum( dhat_k .* z_hat, 3))) + smoothinit;
    res = zeros(video_size(1), video_size(2)*video_size(3));
    for k=1:video_size(3)
        res(:, (k-1)*video_size(2)+1:k*video_size(2)) = Dz(1 + psf_radius(1):end - psf_radius(1), (k-1)*(video_size(2))+ k*psf_radius(1)+ 1 : k*(video_size(2)+psf_radius(1)));
    end
%     res = Dz(1 + psf_radius(1):end - psf_radius(1),1 + psf_radius(2):end - psf_radius(2), : );
    res(res < 0) = 0;
    
    Dz = real(ifft2( sum( dhat .* z_hat, 3)))  + smoothinit;
    res_b = zeros(video_size(1), video_size(2)*video_size(3));
    for k=1:video_size(3)
        res_b(:, (k-1)*video_size(2)+1:k*video_size(2)) = Dz(1 + psf_radius(1):end - psf_radius(1), (k-1)*(video_size(2))+ k*psf_radius(1)+ 1 : k*(video_size(2)+psf_radius(1)));
    end
%     res_b = Dz(1 + psf_radius(1):end - psf_radius(1),1 + psf_radius(2):end - psf_radius(2), : );
    res_b(res_b < 0) = 0;
    
return;

function [MtM, Mtb] = precompute_MProx(b, smoothinit, mask, psf_radius, video_size)
    
    M = zeros(video_size(1) + psf_radius(1), video_size(3)*video_size(2) + psf_radius(1)*(video_size(3)));
    Mtb = zeros(video_size(1) + psf_radius(1), video_size(3)*video_size(2) + psf_radius(1)*(video_size(3)));
    for i=1:video_size(3)
        Mi = padarray(mask(:,(i-1)*video_size(2) + 1 : i*video_size(2) ), psf_radius, 0, 'post');
        M(:, (i-1)*(video_size(2)+psf_radius(1)) + 1 : i*(video_size(2)+psf_radius(1))) = Mi;
        
        Mtbi = padarray(b(:,(i-1)*video_size(2) + 1 : i*video_size(2) ), psf_radius, 0, 'post');
        Mtb(:, (i-1)*(video_size(2)+psf_radius) + 1 : i*(video_size(2)+psf_radius)) = Mtbi;        
    end
    M = padarray(M, psf_radius, 0, 'pre');
    MtM = M .* M;
    Mtb = padarray(Mtb, psf_radius, 0, 'pre') .* M - smoothinit .* M;
    
return;

function [dhat_k, dhat, dhat_flat, dhatTdhat_flat] = precompute_H_hat(kmat, psf, size_x )
% Computes the spectra for the inversion of all H_i

%Precompute PSF
psf_hat = psf2otf(psf, size_x);

%Precompute spectra for H
dhat = zeros( [size_x(1), size_x(2), size(kmat,3)] );
for i = 1:size(kmat,3)  
    dhat(:,:,i)  = psf_hat .* psf2otf(kmat(:,:,i), size_x);
end

%Precompute spectra for H
dhat_k = zeros( [size_x(1), size_x(2), size(kmat,3)] );
for i = 1:size(kmat,3)  
    dhat_k(:,:,i)  = psf2otf(kmat(:,:,i), size_x);
end

%Precompute the dot products for each frequency
dhat_flat = reshape( dhat, size_x(1) * size_x(2), [] );
dhatTdhat_flat = sum(conj(dhat_flat).*dhat_flat,2);

return;


function z_hat = solve_conv_term(dhat, dhatT, dhatTdhat, xi_hat, gammas, size_z)


    % Solves sum_j gamma_i/2 * || H_j z - xi_j ||_2^2
    % In our case: 1/2|| Dz - xi_1 ||_2^2 + rho * 1/2 * || z - xi_2||
    % with rho = gamma(2)/gamma(1)    
  
    
    %Rho
    rho  = gammas(2)/(gammas(1));
    rho2 = gammas(3)/(gammas(1));
    
    %Compute b
    b = dhatT .* repmat( reshape(xi_hat{1}, size_z(1)*size_z(2), 1).', [size(dhat,2),1] ) + rho.*reshape(xi_hat{2}, size_z(1)*size_z(2), size_z(3)).' + rho2.*reshape(xi_hat{3}, size_z(1)*size_z(2), size_z(3)).';
    
    %Invert
    scInverse = repmat( ones(size(dhatTdhat.')) ./ ( rho2.*ones(size(dhatTdhat.')) + rho.*ones(size(dhatTdhat.')) + dhatTdhat.'), [size(dhat,2),1] );
%     x = 1/rho *b - 1/rho * scInverse .* dhatT .* repmat( sum(conj(dhatT).*b, 1), [size(dhat,2),1] );
    x= b.*scInverse;
    %Final transpose gives z_hat
    z_hat = reshape(x.', size_z);

return;

function f_val = objectiveFunction( z, dhat, b, mask, lambda_residual, lambda, psf_radius, video_size )
    
    %Dataterm and regularizer
    Dz = real(ifft2( sum( dhat .* fft2(z), 3)));
    Dz_disp = zeros(video_size(1), video_size(2)*video_size(3));
    for k=1:video_size(3)
        Dz_disp(:, (k-1)*video_size(2)+1:k*video_size(2)) = Dz(1 + psf_radius(1):end - psf_radius(1), (k-1)*(video_size(2))+ k*psf_radius(1)+ 1 : k*(video_size(2)+psf_radius(1)));
    end
%     f_z = lambda_residual * 1/2 * norm( reshape( mask .* Dz(1 + psf_radius(1):end - psf_radius(1),1 + psf_radius(2):end - psf_radius(2),:) - mask .* b, [], 1) , 2 )^2;
    f_z = lambda_residual * 1/2 * norm( reshape( mask .* Dz_disp - mask .* b, [], 1) , 2 )^2;
    g_z = lambda * sum( abs( z(:) ), 1 );
    
    %Function val
    f_val = f_z + g_z;
    
return;

% function prox = ProxTemp(u,theta)
%     prox = zeros(size(u));
%     for t = 1:size(u,3)
%         auxi = u(:,:,t);
%         if (norm(auxi(:)) > theta)
%             prox(:,:,t) = (1-theta/norm(auxi(:))).*auxi;
%         end
%     end
% return;
