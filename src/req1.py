import numpy as np
from PIL import Image

def req1(img: Image.Image) -> Image.Image:
    """Applies the step function from requirement 1"""
    def step(img: Image.Image) -> np.ndarray[int]:
        """Generates the numpy array with the transformation"""
        im = np.array(img)
        output: np.ndarray = np.copy(im) # Copy by value
        def interpolate(value: int) -> int:
            """Will calculate the new value"""
            # I wish I could have come up with something better but
            holder: int = -1
            if value <= 30:
                holder = 10
            elif 30 < value <=60:
                holder = 20
            elif 60 < value <= 90:
                holder = 50
            elif 90 < value <= 120:
                holder = 70
            elif 120 < value <= 160:
                holder = 100
            elif 160 < value <= 190:
                holder = 140
            elif 190 < value <= 220:
                holder = 180
            else:
                holder = 200
            return holder
        it = np.nditer(im, ['multi_index'])
        while not it.finished :
            shape = it.multi_index
            new: int = interpolate(it[0])
            # Apply transformations
            output[shape] = new
            # Iterate
            it.iternext()
        return output
    output: Image.Image = Image.fromarray(step(img = img))
    return output

if __name__ == "__main__":
    holder = req1(Image.open('../figures/first requirement before.png'))
    holder.save('../figures/first requirement after.png')

