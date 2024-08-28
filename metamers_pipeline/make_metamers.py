import torch
import torchaudio
import torchaudio.transforms as T
import sys
import tqdm
import scipy

from speechbrain.inference.speaker import EncoderClassifier
# from speechbrain.inference.speaker import SpeakerRecognition


# add robustness in inputs
sys.path.append('/om2/user/salavill/misc/voice-speech-metamers/metamers_pipeline/robustness')
from attack_steps import L2Step
from custom_synthesis_losses import InversionLossLayer_ecapa

####################################################################
#################### Define relevant variables #####################
####################################################################
# Audio
audio = '/om2/user/amagaro/voice-speech-metamers/metamers_pipeline/kell2018/metamers/psychophysics_wsj400_jsintest_inversion_loss_layer_RS0_I3000_N8/0_SOUND_million/orig.wav'
sr = 16000
noise_scale = .0000001

# loss param
normalize_loss = False

# metamer generation
eps = 100000
step_size = 1

random_start = 0 #UPDATE: IDK
do_tqdm = 1
use_best = 0
est_grad = None

targeted = 1
m = -1 if targeted else 1 

iterations = 3000 # in the pipeline Janelle uses 3000
total_iterations = 8
return_image = 1


####################################################################
##################### Define Metamer Function ######################
####################################################################

# Main function for making adversarial examples
def get_adv_examples(x):
    # Random start (to escape certain types of gradient masking)
    if random_start:
        x = step.random_perturb(x)

    iterator = range(iterations)
    if do_tqdm: iterator = tqdm(iterator)

    # Keep track of the "best" (worst-case) loss and its
    # corresponding input
    best_loss = None
    best_x = None

    # A function that updates the best loss and best input
    def replace_best(loss, bloss, x, bx):
        if bloss is None:
            bx = x.clone().detach()
            bloss = loss.clone().detach()
        else:
            replace = m * bloss < m * loss
            bx[replace] = x[replace].clone().detach()
            bloss[replace] = loss[replace]

        return bloss, bx

    # PGD iterates
    for _ in iterator:
        # main metamer generation
        x = x.clone().detach().requires_grad_(True)
        losses, out = calc_loss(model, step.to_image(x), target)
        assert losses.shape[0] == x.shape[0], \
                'Shape of losses must match input!'

        loss = torch.mean(losses)

        if step.use_grad:
            if est_grad is None:
                grad, = torch.autograd.grad(m * loss, [x])
            # else:
            #     f = lambda _x, _y: m * calc_loss(step.to_image(_x), _y)[0]
            #     grad = helpers.calc_est_grad(f, x, target, *est_grad)
        else:
            grad = None

        with torch.no_grad():
            args = [losses, best_loss, x, best_x]
            best_loss, best_x = replace_best(*args) if use_best else (losses, x)

            x = step.step(x, grad)
            x = step.project(x)
            if do_tqdm: iterator.set_description("Current loss: {l}".format(l=loss))

    # Save computation (don't compute last loss) if not use_best
    if not use_best: 
        ret = x.clone().detach()
        return step.to_image(ret) if return_image else ret

    losses, _ = calc_loss(model, step.to_image(x), target)
    args = [losses, best_loss, x, best_x]
    best_loss, best_x = replace_best(*args)
    return step.to_image(best_x) if return_image else best_x



####################################################################
####################### Generate Metamer ###########################
####################################################################

# load in sound
signal, fs = torchaudio.load(audio)
print(f'Generating metamer for sound: {audio}')
if fs != sr:
    # make sure to resample to appropriate frequency
    print('resampling audio')
    resampler = T.Resample(fs, sr, dtype=signal.dtype)
    signal = resampler(signal)

# load in model 
model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
print('Loaded in ECAPA model')

# Get target embedding by running signal through model
target = model.encode_batch(signal)[0]

# initialize random noise
im_n_initialized = [(torch.randn_like(signal)) * noise_scale][0]

# get loss function
calc_loss = InversionLossLayer_ecapa(normalize_loss = normalize_loss)

# get min and max value from signal for step function
dataset_min_value = signal.min()
dataset_max_value = signal.max()
# define step function 
step = L2Step(eps=eps, orig_input=signal, step_size=step_size,
            min_value=dataset_min_value, max_value=dataset_max_value)

# initialize a metamer
print('Generating metamer')
xadv = get_adv_examples(im_n_initialized)
this_loss = calc_loss(model, xadv, target.clone())

# iterate through metamer and print every 10 times
for i in range(total_iterations):
    # iterate I times and cut step size in half each time
    im_n = xadv
    step.step_size = step.step_size/2
    # get metamer
    xadv = get_adv_examples(im_n)
    # calculate new loss
    this_loss = calc_loss(model, xadv, target.clone())
    
    # print information
    print(f'Iteration: {i}, Loss: {this_loss}')

# save out current metamer
scipy.io.wavfile.write(filename = f'metamers/{audio}_final.wav', rate=sr, data = xadv)