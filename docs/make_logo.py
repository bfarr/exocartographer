from PIL import Image, ImageFont, ImageDraw

logo = Image.new("RGBA", (600, 200), (255, 255, 255))
draw = ImageDraw.Draw(logo)
draw.text((10, 0), "exocartographer", (0, 0, 0), font=font)
logo.save('_static/exocartographer.png')
