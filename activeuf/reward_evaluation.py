import subprocess
import sys

def run_rewardbench():
    command = [
        "rewardbench",
        "--model=TrainedRewardModel",
        "--dataset=allenai/ultrafeedback_binarized_cleaned",
        "--split=test_gen",
        "--chat_template=raw"
    ]
    
    
    try:
        subprocess.run(command, check=True, stdout=sys.stdout, stderr=sys.stderr, text=True)
    except subprocess.CalledProcessError as e:
        print("Command failed with return code", e.returncode)

### Debugging
"""
def debug_rewardbench():
    from rewardbench.rewardbench import main
    sys.argv = [
        "rewardbench",
        "--model=TrainedRewardModel", #OpenAssistant/reward-model-deberta-v3-base"
        "--dataset=allenai/ultrafeedback_binarized_cleaned", 
        "--split=test_gen",
        "--chat_template=raw"
    ]
    print(sys.argv)
    main()
"""


if __name__ == "__main__":
    run_rewardbench()
