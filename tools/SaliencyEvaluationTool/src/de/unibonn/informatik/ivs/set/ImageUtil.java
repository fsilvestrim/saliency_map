
package de.unibonn.informatik.ivs.set;


import java.io.*;
import java.awt.*;
import java.awt.image.*;
import javax.imageio.*;


/**
 * Provides supporting methods for handling images
 *
 * @author		Bernd Wendt
 * @version		2011.0314
 *
 */
public class ImageUtil
{

	/**
	 * Converts an image of any type to a grey image
	 *
	 * @param image the image to grey
	 * @return      the greyed image
	 */
	public static BufferedImage greyImage(BufferedImage image)
	{
		BufferedImage result = new BufferedImage(image.getWidth(), image.getHeight(), BufferedImage.TYPE_BYTE_GRAY);
		Graphics2D graphics2D = result.createGraphics();
		graphics2D.drawImage(image, 0, 0, result.getWidth(), result.getHeight(), null);
		graphics2D.dispose();

		return result;
	}


	/**
	 * Scales an image to a specified size.<br>
	 * The method used is bicubic interpolation.
	 *
	 * @param image  image to scale
	 * @param width  target width
	 * @param height target height
	 *
	 * @return       scaled image
	 */
	public static BufferedImage scaleImage(BufferedImage image, int width, int height)
	{
		BufferedImage result = null;

		if (image.getWidth() != width || image.getHeight() != height)
		{
			// Create new (blank) image of required size
			result = new BufferedImage(width, height, image.getType());

			// Paint scaled version of image to new image
			Graphics2D graphics2D = result.createGraphics();
			graphics2D.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC);
			graphics2D.drawImage(image, 0, 0, width, height, null);

			// clean up
			graphics2D.dispose();
		}
		else result = image;

		return result;
	}


	/**
	 * Loads an image from a file.<br>
	 * If loading fails (for example because file does not exist) it returns null.
	 *
	 * @param imageFileName name/path of the image file to load
	 * @return              the loaded image
	 */
	public static BufferedImage loadImage(String imageFileName)
	{
		return loadImage(new File(imageFileName));
	}


	/**
	 * Loads an image from a file.<br>
	 * If loading fails (for example because file does not exist) it returns null.
	 *
	 * @param imageFile file that contains an image
	 * @return          the loaded image
	 */
	public static BufferedImage loadImage(File imageFile)
	{
		BufferedImage result = null;

		try
		{
			result = ImageIO.read(imageFile);
		}
		catch (IOException e)
		{
			System.err.println("Could not load image '"+imageFile.getAbsolutePath()+"': "+e.toString());
		}

		return result;
	}


	/**
	 * Saves an image to a file in PNG format.<br>
	 * The appropriate file ending is added automatically if necessary.
	 *
	 * @param image    the image to save
	 * @param fileName the name of the file to save image to
	 */
	public static void saveImageAsPng(BufferedImage image, String fileName)
	{
		try
		{
			if (!fileName.endsWith(".png")) fileName += ".png";
			File file = new File(fileName);

			ImageIO.write(image, "PNG", file);
		}
		catch (Exception e)
		{
			System.err.println("Error saving image '"+fileName+"': "+e.toString());
		}
	}
}
