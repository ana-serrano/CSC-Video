function sampling3D = create_mask(M, N, T, psize,bump_size);


p = psize(1);
q = psize(2);
bump_start = zeros(M,N);

for i = 1:(M-p+1)
    fprintf('Creating mask: Row %d\n',i);
    if (i==(M-p+1))
        disp('Creating mask: Last row')
    end
    j=1;
    while(j<=(N-q+1))
        count = 0;
        repeat = 1;
        patch = bump_start((i:(i+p-1)),(j:(j+q-1)));
        suma = 0;
        %Which elements of the patch are zero 
        zeroElements = (patch==0);
        %As many new random element as elements were zero
        news = sum(zeroElements(:));
        %If any element is zero
        if (news>0)
            %Save indices of such elements
            index = find(patch==0);
            %Added elements to patch => Repear until conditions matched
            while (repeat == 1)
                  count = count+1;
                  if (count>500)
                     j=1; break;%If we get stuck, restart the row
                  end
                  repeat = 0;
                  add = randi([-1 T],news,1);
                  for pp = 1:length(add)
                      for qq = 1:bump_size
                        if (add(pp)==T-qq+1)
                            add(pp) = T-bump_size+1;
                        end
                        if (add(pp)==1-qq+1)
                            add(pp) = 1;
                        end
                      end
                  end
                  patch(index) = add;  
                  for ii = 1:T
                      aux = (patch==ii);
                      suma = sum(aux(:));
                      for jj = 1:(bump_size-1)
                          aux = (patch == (ii-jj));
                          suma = suma + sum(aux(:));
                      end
                      if (suma <1)
                          repeat = 1;
                          fprintf('Creating mask: Repeat Row %d Column %d\n',i,j)
                      end
                  end
                  suma = 0;
            end
        end
        if (count<=500)
            bump_start((i:(i+p-1)),(j:(j+q-1))) = patch;
            j=j+1;  
        else
            bump_start(i+p-1,:) = 0;
        end
    end
    
end


%Create 3D shutter matrix

sampling3D = zeros(M,N,T);

for k = 1:T
    frame = sampling3D(:,:,k);
    index = find(bump_start==k);
    frame(index) = 1;
    for i = 1:(bump_size-1)
        index = find(bump_start==(k-i));
        frame(index) = 1;
    end
    sampling3D(:,:,k) = frame;
end

%Check

% for i = 1:size(sampling3D,1);
%     for j = 1:size(sampling3D,2)
%       Msuma(i,j) = sum(sampling3D(i,j,:));
%     end
% end

        