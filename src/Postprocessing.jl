export removeOversampling!

function removeOversampling!(r::ReconParams{<:AbstractKdata, <:AbstractTrajectory, ImgData{T}}) where T<:AbstractFloat
    @debug("Remove oversampling.")
    img = removeOversampling(r.imgData.signal, r.scanParameters[:encodingSize])
    r.imgData.signal = img
    append!(r.performedMethods, [nameof(var"#self#")])
end

function removeOversampling(img, encodingSize) 
    # Crop oversampling
    center = div.(size(img)[1:3], 2)
    # cropInd_x = floor(Int64, 1/2*(size(img, 1)-encodingSize[1]))
    # cropInd_y = floor(Int64, 1/2*(size(img, 2)-encodingSize[2]))
    # cropInd_z = floor(Int64, 1/2*(size(img, 3)-encodingSize[3]))
    # return Array(selectdim(selectdim(selectdim(img, 1, cropInd_x+1:size(img, 1)-cropInd_x), 
    #                            2, cropInd_y+1:size(img, 2)-cropInd_y), 
    #                 3, cropInd_z+1:size(img, 3)-cropInd_z))
    return Array(selectdim(selectdim(selectdim(img, 1, center[1]-div(encodingSize[1], 2)+1:center[1]-div(encodingSize[1],2)+encodingSize[1]), 
                                     2, center[2]-div(encodingSize[2], 2)+1:center[2]-div(encodingSize[2], 2)+encodingSize[2]), 
                           3, center[3]-div(encodingSize[3], 2)+1:center[3]-div(encodingSize[3], 2)+encodingSize[3]))
end