from tkinter import filedialog
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

def fun_image(image, f_w, c):
    n_row, n_col = image.shape
    image = image.copy()
    for i in range(n_row):
        for j in range(n_col):
            x = image[i][j] / 255
            image[i][j] = np.floor(f_w(x, c=c) * 255)
    return image

def gamma_operation(gamma, image):
    n_row, n_col = image.shape
    image = image.copy()
    for i in range(n_row):
        for j in range(n_col):
            x = image[i][j] / 255
            if x > 0:
                image[i][j] = np.floor(x ** gamma * 255)
            else:
                image[i][j] = 0
    return image

def f1(x, c):
    if c > 0:
        return ((1 / (2 * np.arctan(c * np.tan(1 / 2)))) *
                np.arctan(c * 2 * np.tan(1 / 2) * (x - 1 / 2)) + 1 / 2)
    elif c == 0:
        return x
    else:
        return ((1 / (2 * (-c) * np.tan(1 / 2))) *
                np.tan(2 * np.arctan((-c) * np.tan(1 / 2)) * (x - 1 / 2)) + 1 / 2)

def do_mask_on_image(im, kernel):
    im = np.double(im)
    n_row_kernel = kernel.shape[0]
    extension_image = np.pad(im, n_row_kernel // 2, mode='constant')
    image_blur = im.copy()

    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            pixel_new_value = 0
            for m in range(n_row_kernel):
                for k in range(n_row_kernel):
                    weight = kernel[m][k]
                    neighbor = extension_image[i + m][j + k]
                    pixel_new_value += weight * neighbor
            image_blur[i][j] = pixel_new_value

    return image_blur

def scale_lap(im):
    min_val = np.min(im)
    scale_image = im - min_val
    scale_image = scale_image * (510 / np.max(scale_image))
    return scale_image - 255

def rotation(im, teta_st):
    angle = np.radians(teta_st)
    cosine = np.cos(angle)
    sine = np.sin(angle)
    mat_r_o = np.array([[cosine, sine], [-sine, cosine]])

    new_height = int(abs(sine) * im.shape[1] + abs(cosine) * im.shape[0])
    new_width = int(abs(cosine) * im.shape[1] + abs(sine) * im.shape[0])
    original_centre_height = (im.shape[0] - 1) / 2
    original_centre_width = (im.shape[1] - 1) / 2
    original_center = np.array([[original_centre_width], [original_centre_height]])
    new_centre_height = (new_height - 1) / 2
    new_centre_width = (new_width - 1) / 2
    new_center = np.array([[new_centre_width], [new_centre_height]])
    new_im = np.zeros((new_height, new_width))

    for i in range(new_height):
        for j in range(new_width):
            x = np.array([[j], [i]])
            y = np.round((mat_r_o @ (x - new_center)) + original_center)
            if int(y[1][0]) in range(im.shape[0]) and int(y[0][0]) in range(im.shape[1]):
                new_im[i][j] = im[int(y[1][0])][int(y[0][0])]

    return new_im

if __name__ == '__main__':
    # dla image1 nalezy zmienic sciezke pliku w tym miejscu
    image1 = cv2.imread("C:\\Users\\klaud\\Downloads\\zdj_do_tf\\wyjazd1.jpg",
                        cv2.IMREAD_GRAYSCALE)
    # aby moc skorzystac z programu, na sciezke do dowolnego zdjecia z komputera uzytkownika
    image1 = np.double(image1)

    # skalowanie, jesli jest wymagane
    if (image1.shape[0] > 500) or (image1.shape[1] > 500):
        kk = 500 / max(image1.shape[0], image1.shape[1])
        original_height, original_width = image1.shape
        new_height1, new_width1 = (int(original_height * kk), int(original_width * kk))
        sc_image = np.zeros((new_height1, new_width1))

        for ii in range(new_height1):
            for jj in range(new_width1):
                xx = np.round(np.array([[jj], [ii]]) * (1 / kk))
                if (int(xx[1][0]) in range(image1.shape[0]) and
                        int(xx[0][0]) in range(image1.shape[1])):
                    sc_image[ii][jj] = image1[int(xx[1][0])][int(xx[0][0])]
        image1 = sc_image

    # maski
    gaussian_kernel_3x3 = 1 / 16 * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    laplacian_kernel_3x3 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

    # zasadnicza konstrukcja pojawiajacego sie okna
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25, right=0.75)
    axcolor = 'lightblue'

    g_0 = 1
    c_0 = 0
    teta_0 = 0

    current_image = image1
    ax.imshow(current_image, cmap='gray')
    ax.set_xticks([]), ax.set_yticks([])

    axs = plt.axes((0.25, 0.1, 0.5, 0.03), facecolor=axcolor)
    axgamma = plt.axes((0.25, 0.15, 0.5, 0.03), facecolor=axcolor)
    axteta = plt.axes((0.25, 0.05, 0.5, 0.03), facecolor=axcolor)

    sc = Slider(axs, 'Kontrast($c$)', -20, 20, valinit=c_0)
    sgamma = Slider(axgamma, 'Jasnosc($\\gamma$)', 0.1, 5, valinit=g_0)
    steta = Slider(axteta, 'Kat obrotu($\\theta$)', 0, 360, valinit=teta_0)

    def update(val):
        global current_image
        g = sgamma.val
        c = sc.val
        teta = steta.val
        current_image = image1.copy()
        current_image = fun_image(current_image, f1, c)
        current_image = rotation(current_image, teta)
        current_image = gamma_operation(g, current_image)
        ax.imshow(current_image, cmap='gray')
        fig.canvas.draw_idle()

    sc.on_changed(update)
    sgamma.on_changed(update)
    steta.on_changed(update)

    sharpax = plt.axes((0.8, 0.15, 0.1, 0.03))
    blurax = plt.axes((0.8, 0.11, 0.1, 0.03))
    resetax = plt.axes((0.8, 0.07, 0.1, 0.03))
    saveax = plt.axes((0.8, 0.03, 0.1, 0.03))

    button_sharp = Button(sharpax, 'Wyostrzenie', color=axcolor, hovercolor='0.5')
    button_blur = Button(blurax, 'Rozmywanie', color=axcolor, hovercolor='0.9')
    button_reset = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
    button_save = Button(saveax, 'Zapisz obraz', color=axcolor, hovercolor='0.975')

    def apply_blur(event):
        global current_image
        current_image = do_mask_on_image(do_mask_on_image(current_image, gaussian_kernel_3x3),
                                         gaussian_kernel_3x3)
        ax.imshow(current_image, cmap='gray')
        fig.canvas.draw_idle()

    def apply_sharp(event):
        global current_image
        lap_mask = do_mask_on_image(current_image, laplacian_kernel_3x3)
        c = -1
        current_image = np.clip(current_image + c * scale_lap(lap_mask), 0, 255)
        ax.imshow(current_image, cmap='gray')
        fig.canvas.draw_idle()

    def reset(event):
        sc.reset()
        sgamma.reset()
        steta.reset()
        global current_image
        current_image = image1
        ax.imshow(current_image, cmap='gray')
        fig.canvas.draw_idle()

    def save_image(event):
        global current_image
        file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG files", "*.png"),
                                                            ("JPEG files", "*.jpg"),
                                                            ("All files", "*.*")])
        if file_path:
            cv2.imwrite(file_path, current_image)

    button_blur.on_clicked(apply_blur)
    button_sharp.on_clicked(apply_sharp)
    button_reset.on_clicked(reset)
    button_save.on_clicked(save_image)

    def resize_fonts(event=None):
        for button in [button_sharp, button_blur, button_reset, button_save]:
            bbox = button.ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            width, height = bbox.width, bbox.height
            fontsize = min(width, height) * 35
            button.label.set_fontsize(fontsize)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('resize_event', resize_fonts)
    resize_fonts()

    plt.show()
