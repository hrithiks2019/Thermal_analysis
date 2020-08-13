import os
import cv2
import json
import time
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from reportlab.lib import colors
from Temp_Pixel import get_max_temp
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from Object_Recognition import get_object
from Temp_Pixel import max_temp_pixel_value
from reportlab.pdfbase.ttfonts import TTFont


def display_multiple_img(images, rows, cols):
    figure, ax = plt.subplots(nrows=rows, ncols=cols)
    for ind, title in enumerate(images):
        ax.ravel()[ind].imshow(images[title])
        ax.ravel()[ind].set_title(title)
        ax.ravel()[ind].set_axis_off()
    figure.set_size_inches(4, 4)
    plt.tight_layout()
    plt.savefig('main_data_set/temp_fig.png')


def gen_pdf_file(filename, data):
    title, subtitle = 'THERMAL ANALYSIS', 'System Study'
    image = 'main_data_set/temp_fig.png'
    pdf = canvas.Canvas(filename)
    pdf.setTitle(filename)
    pdfmetrics.registerFont(TTFont('abc', 'Support/SakBunderan.ttf'))
    pdf.setFont('abc', 36)
    pdf.drawCentredString(300, 770, title)
    pdf.setFillColorRGB(0, 0, 255)
    pdf.setFont("Courier-Bold", 24)
    pdf.drawCentredString(290, 720, subtitle)
    pdf.line(30, 710, 550, 710)
    text = pdf.beginText(40, 680)
    text.setFont("Courier", 16)
    text.setFillColor(colors.red)
    for line in data:
        text.textLine(line)
    pdf.drawText(text)
    pdf.setFillColorRGB(0, 0, 255)
    pdf.setFont("Courier-Bold", 24)
    pdf.drawCentredString(300, 450, 'Image Analysis')
    pdf.line(30, 440, 550, 440)
    pdf.drawInlineImage(image, 90, 0)
    os.remove('main_data_set/temp_fig.png')
    pdf.save()


normal_image_path = "main_data_set/test_normal_wire.jpg"
thermal_image_path = "main_data_set/test_thermal_wire.jpg"

paramter_json = open('Support/parameter.json', "r")
paramters = json.loads(paramter_json.read())
filename = 'Report_' + str(datetime.now().strftime("%d-%m-%Y_%H%M%S")) + '.pdf'

object_dxi = 'UNKNOWN'
final_report = ['       PARAMETERS               ANALYSED-OUTPUT        ',
                '                                                       ']
k_constant = 100

print('[INFO] Process Running: Object Detection ')
# noinspection PyRedeclaration
object_dxi = get_object(thermal_image_path)
print('[INFO] Process Running: Getting Temperature Reading at the Hottest Area of the Object ')
maxi_temp = get_max_temp(thermal_image_path)
print('[INFO] Process Running: Setting The Upper Scale Spectrum for Thermal Analysis ')
upper = max_temp_pixel_value(thermal_image_path)
print('[INFO] Proceeding To Analyse the Two Images')

lower = [upper[0] - k_constant, upper[1] - k_constant, upper[2] - k_constant]
lower = np.array(lower, dtype="uint8")
upper = np.array(upper, dtype="uint8")
normal_image = cv2.imread(normal_image_path)
thermal_image = cv2.imread(thermal_image_path)
hsv = cv2.cvtColor(thermal_image, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(thermal_image, lower, upper)
output = cv2.bitwise_and(thermal_image, thermal_image, mask=mask)
ret, thresh = cv2.threshold(mask, 40, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
if len(contours) != 0:
    cv2.drawContours(output, contours, -1, 255, 3)
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    org = (x - 5, y - 5)
    cv2.rectangle(normal_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(normal_image, 'Object: ' + str(object_dxi), org, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3,
                cv2.LINE_AA)
    cv2.rectangle(thermal_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(thermal_image, 'Hottest Area', org, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(output, 'Hotspot', org, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.rectangle(hsv, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.putText(hsv, 'Heat Focused Site', org, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

thermal_image = cv2.cvtColor(thermal_image, cv2.COLOR_BGR2RGB)
normal_image = cv2.cvtColor(normal_image, cv2.COLOR_BGR2RGB)
hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2RGB)
output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
results = {"Thermal Image": thermal_image, "Color Image": normal_image,
           "HSV Analysed Image": hsv, "Thermal_hotspot Image": output}
print('[INFO] ANALYSING ACQUIRED DATA')

final_report.append('OBJECT CLASSIFICATION  :  {0}'.format(str(object_dxi)))
final_report.append('OBJECT TEMPERATURE     :  {0} celcius'.format(maxi_temp))
final_report.append('NOMINAL TEMPERATURE    :  {0} Celcius'.format(paramters[object_dxi]['normal_temp']))
final_report.append('TOLERANCE TEMPERATURE  :  {0} Celcius'.format(paramters[object_dxi]['max_temp']))
if maxi_temp > paramters[object_dxi]['max_temp']:
    final_report.append('RISK ESTIMATION        :  HIGH RISK')
    final_report.append('CURRENT STATE          :  {0} Deg Higher than Tolerance'.
                        format(round(maxi_temp - (paramters[object_dxi]['max_temp']), 2)))
    final_report.append('ANALYTICAL OPINION     :  This component Should Be Taken Offline and '
                        'Analysed by Technician Immediately.')
    final_report.append('LONGEVITY ANALYSIS     :  0-1 Months')
elif maxi_temp < paramters[object_dxi]['max_temp']:
    final_report.append('RISK ESTIMATION        :  MEDIUM RISK')
    final_report.append('CURRENT STATE          :  {0} Deg Higher than Normal'.
                        format(round(maxi_temp - (paramters[object_dxi]['normal_temp']), 2)))
    final_report.append('ANALYTICAL OPINION     :  TECHNICIAN SHOULD TAKE A LOOK')
    final_report.append('LONGEVITY ANALYSIS     :  12-18 Months                 ')
else:
    final_report.append('RISK ESTIMATION        : LOW RISK')
    final_report.append('CURRENT STATE: It is Well Within Limits')
    final_report.append('ANALYTICAL OPINION: NONE')
    final_report.append('LONGEVITY ANALYSIS: 2-3 YEARS')


display_multiple_img(results, 2, 2)
gen_pdf_file(filename, final_report)
print('[INFO] Analysis Completed')
print('[INFO] Report Has Been Generated saving as : {0}'.format('Report_' + str(
        datetime.now().strftime("%d-%m-%Y_%H%M%S")) + '.pdf'))
print('[INFO] Now Opening the Report PDF File')
time.sleep(2)
os.system(filename)
