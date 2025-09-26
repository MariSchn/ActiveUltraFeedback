

from accelerate import Accelerator

from activeuf.acquisition_function import init_acquisition_function
from activeuf.loop.arguments import get_args


if __name__ == "__main__":
    accelerator = Accelerator()
    args = get_args()
    
    acquisition_function = init_acquisition_function(
        args.acquisition_function,
        # seed = args.seed, # TODO: pass the global seed here? or maybe we should set it as an env variable that is accessible everywhere. this seems very hacky
        **args.acquisition_function_config.get(args.acquisition_function, {})
    )
    
