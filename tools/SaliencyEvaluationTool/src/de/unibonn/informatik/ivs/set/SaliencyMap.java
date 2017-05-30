
package de.unibonn.informatik.ivs.set;


import java.io.*;
import java.awt.image.*;


/**
 * Represents a saliency map for an image.<br>
 * <br>
 * A saliency map is represented by a greyscale image and always refers to an image
 * for which the saliency map was calculated.
 * Thus a SaliencyMap instance is always created with a greyscale saliency image.
 * As some algorithms create saliency maps that differ in size from the image they refer to,
 * the size of that image must be provided to scale the saliency map internally.<br>
 * As a result all methods that return a representation of the saliency map
 * ({@link #getMap()} and {@link #getMapValues()}) return a representaion that fits to the size
 * of the image the saliency map refers to.
 *
 * @author		Bernd Wendt
 * @version		2011.0314
 *
 */
public class SaliencyMap
{

	/**
	 * Number of greyscales used by saliency maps
	 */
	public static final int NUM_GREYSCALES = 256;

	/**
	 * Name of the image saliency map refers to
	 */
	String imageName;

	/**
	 * The saliency map as an greyscale image
	 */
	BufferedImage saliencyMap;


	/**
	 * Constructs from a saliency map image
	 *
	 * @param fileName file name of saliency image
	 * @param width    width of image saliency map refers to
	 * @param height   height of image saliency map refers to
	 */
	public SaliencyMap(String fileName, int width, int height)
	{
		loadSaliencyMap(fileName, width, height);
	}


	/**
	 * Constructs from a saliency map image
	 *
	 * @param file   file containing saliency image
	 * @param width  width of image saliency map refers to
	 * @param height height of image saliency map refers to
	 */
	public SaliencyMap(File file, int width, int height)
	{
		loadSaliencyMap(file, width, height);
	}


	/**
	 * Loads saliency map image.<br>
	 * If the saliency image differs from the size of the image it refers to,
	 * the saliency image is scaled up/down.
	 *
	 * @param fileName file name of saliency image
	 * @param width    width of image saliency map refers to
	 * @param height   height of image saliency map refers to
	 */
	private void loadSaliencyMap(String fileName, int width, int height)
	{
		File file = new File(fileName);
		loadSaliencyMap(file, width, height);
	}


	/**
	 * Loads saliency map image.<br>
	 * If the saliency image differs from the size of the image it refers to,
	 * the saliency image is scaled up/down.
	 *
	 * @param file   file containing saliency image
	 * @param width  width of image saliency map refers to
	 * @param height height of image saliency map refers to
	 */
	private void loadSaliencyMap(File file, int width, int height)
	{
		saliencyMap = ImageUtil.loadImage(file);

		if (saliencyMap != null)
		{
			// image name is file name without ending
			imageName = FileUtil.getFileNameWithoutEnding(file.getName());

			if (saliencyMap.getType() != BufferedImage.TYPE_BYTE_GRAY) saliencyMap = ImageUtil.greyImage(saliencyMap);

			if (saliencyMap.getWidth() != width || saliencyMap.getHeight() != height) saliencyMap = ImageUtil.scaleImage(saliencyMap, width, height);
		}
	}


	/**
	 * Returns an greyscale image that represents the saliency map.
	 *
	 * @return saliency image
	 */
	public BufferedImage getMap()
	{
		return saliencyMap;
	}


	/**
	 * Returns the width of the saliency map.<br>
	 * It is the same as the width of the image the saliency map refers to.<br>
	 * <br>
	 * If saliency map is not defined, return value is -1.
	 *
	 * @return width of saliency map
	 */
	public int getWidth()
	{
		if (saliencyMap != null) return saliencyMap.getWidth();
		return -1;
	}


	/**
	 * Returns the width of the saliency map.<br>
	 * It is the same as the width of the image the saliency map refers to.<br>
	 * <br>
	 * If saliency map is not defined, return value is -1.
	 *
	 * @return width of saliency map
	 */
	public int getHeight()
	{
		if (saliencyMap != null) return saliencyMap.getHeight();
		return -1;
	}


	/**
	 * Returns the values of the map as one-dimensional array of integers.
	 * Values are in range [0; 255] and represent the saliency map column by column.
	 *
	 * @return array containing the map's values
	 */
	public int[] getMapValues()
	{
		if (saliencyMap == null) return null;

		int[] result = new int[saliencyMap.getWidth()*saliencyMap.getHeight()];
		int i=0;

		for (int x=0; x<saliencyMap.getWidth(); x++)
		{
			for (int y=0; y<saliencyMap.getHeight(); y++)
			{
				result[i++] = saliencyMap.getRGB(x, y) & 0xFF;
			}
		}

		return result;
	}


	/**
	 * Returns a binary image of all saliency values above or equal to a threshold.
	 *
	 * @param binaryThreshold minimum for values to be set in binary image
	 *
	 * @return                binary image
	 */
	BufferedImage getBinaryMap(int binaryThreshold)
	{
		BufferedImage result = new BufferedImage(saliencyMap.getWidth(), saliencyMap.getHeight(), BufferedImage.TYPE_BYTE_BINARY);

		for (int x=0; x<saliencyMap.getWidth(); x++)
		{
			for (int y=0; y<saliencyMap.getHeight(); y++)
			{
				if ((saliencyMap.getRGB(x, y) & 0xFF) >= binaryThreshold) result.setRGB(x, y, 0xFFFFFFFF);
			}
		}

		return result;

	}
}
