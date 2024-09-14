function motionStatesClusteringStackOfStars(bCurveKz0::Array{T, 1}, bCurve::Array{T, 1}; 
        numClusters::Int=5, method::Symbol=:relDisplacement, 
        numKz::Int) where T
    dist = pairwise(Euclidean(), bCurveKz0)
    if method == :kmedoids
        # Perform clustering
        k = kmedoids(dist, numClusters)
        medoidsCenter = k.medoids[sortperm(bCurveKz0[k.medoids])]
        # Assign intra-shot profiles
        dist2 = pairwise(Euclidean(), bCurve, bCurveKz0[medoidsCenter])
        assignments = argmin.(eachrow(dist2))
    elseif method == :relDisplacement
        cluster_range = 1/numClusters # motion curve should be within this interval
        medoidsCenter = collect((cluster_range/2):cluster_range:(1-cluster_range/2)) # motion curve should be sorted along these interval centers
        dist2 = pairwise(Euclidean(), bCurve, medoidsCenter)
        assignments = argmin.(eachrow(dist2))
    elseif method == :equal_spokes_no
        no_spokes = length(bCurveKz0) # spoke groups per Kz0 
        no_spokes_per_cluster = round(Int,(no_spokes / numClusters))
        bcurveKz0_sorted_idx = sortperm(bCurveKz0)
        # bcurve_sorted_idx = Vector(reshape(transpose(repeat(bcurveKz0_sorted_idx, 1, numKz)), length(bcurveKz0_sorted_idx)*numKz))
        assignments_kz0 = Array{Int16}(undef,no_spokes,1)
        medoidsCenter = Array{Float64}(undef,numClusters,1)

        if numClusters == 1
            assignments_kz0[:] .= 1
        else
            for (class, idx) in enumerate(collect(1:no_spokes_per_cluster:no_spokes)[1:end-1])
                idx_end = min((idx + no_spokes_per_cluster), no_spokes)
                for j in collect(bcurveKz0_sorted_idx[idx:idx_end])
                    assignments_kz0[j,1] = class
                end
                medoidsCenter[class, 1] = mean(hcat([bCurve[i] for i in collect(bcurveKz0_sorted_idx[idx:idx_end])]))
            end
        end
        # for (class, idx) in enumerate(collect(1:no_spokes_per_cluster:no_spokes)[1:end-1])
        #     idx_end = min((idx + no_spokes_per_cluster), no_spokes)
        #     for j in collect(bcurveKz0_sorted_idx[idx:idx_end])
        #         assignments_kz0[j,1] = class
        #     end
        #     medoidsCenter[class, 1] = mean(hcat([bCurve[i] for i in collect(bcurveKz0_sorted_idx[idx:idx_end])]))
        # end

        ## unfold kz0 spokes to all spokes
        assignments = Vector(reshape(transpose(repeat(assignments_kz0, 1, numKz)), length(assignments_kz0)*numKz))
        
        ## test:
        # using StatsBase
        # countmap(assignments)
    end
    return assignments, medoidsCenter
end

function plotBcurve(bCurve::Vector{T}, filename::String="."; assignments::Vector{Int}=Vector{Int}(), 
        numClusters::Int=0) where T
    plot(bCurve, label="motion curve")
    if numClusters > 0
        for i in 1:numClusters
            plot!((0:length(bCurve)-1)[assignments.==i], 
                bCurve[assignments.==i], seriestype=:scatter, label="state "*string(i))
        end
    end
    # plot!(legend=:outerbottom, legendcolumns=3)
    xlabel!("shot number")
    ylabel!("relative displacement (a.u.)")
    # ylims!(minimum(bCurve),maximum(bCurve)) 
    # xlims!(0,100)
    splittedFilename = split(filename, ".")
    savefig(join(splittedFilename[1:end-1], ".") * "_motionCurve.png")
end