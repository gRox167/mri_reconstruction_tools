using PythonCall
pyimport(".mapvbvd")
module recon
    mutable struct AbstractReconArgs
        dat_file_location::String
    end
    mutable struct CAPTURE_VarW_NQM_DCE_PriorInj <: AbstractReconArgs
        phase_num::String
        percentW::Int
    end
end

function args_init(args::recon.AbstractReconArgs, )
    
end