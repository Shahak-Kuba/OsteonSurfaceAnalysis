using Test
using OsteonSurfaceAnalysis

@testset "Utils.fill_internal_zeros" begin
    M = [0 0 1 0 0;
         0 1 0 1 0;
         0 1 0 1 0;
         0 1 0 1 0;
         0 0 1 0 0]
    filled = OsteonSurfaceAnalysis.Utils.fill_internal_zeros(M)
    @test sum(filled) > sum(M)  # got filled
end
