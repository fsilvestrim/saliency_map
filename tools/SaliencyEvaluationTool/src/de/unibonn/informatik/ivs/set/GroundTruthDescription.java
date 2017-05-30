
package de.unibonn.informatik.ivs.set;


import java.util.*;
import java.awt.*;


/**
 * Represents a description for a ground truth of salient objects in an associated image.<br>
 * <br>
 * The representation consists of a set of rectangles determining salient objects in the associated image.<br>
 * For more detailed information see<br>
 * <code>Tie Liu, Jian Sun, Nan-Ning Zheng, Xiaoou Tang, Heung-Yeung Shum. "Learning to Detect A Salient Object". IEEE CVPR 2007.</code><br>
 * <a href="http://research.microsoft.com/en-us/um/people/jiansun/SalientObject/salient_object.htm">http://research.microsoft.com/en-us/um/people/jiansun/SalientObject/salient_object.htm</a>
 *
 * @author		Bernd Wendt
 * @version		2011.0314
 */
public class GroundTruthDescription
{

	/**
	 * Name of the associated image.
	 */
	String imageName;

	/**
	 * Path of the file of the associated image (without file name)
	 */
	String imagePath;

	/**
	 * Size of the associated image
	 */
	Dimension imageSize;

	/**
	 * Rectangles representing salient objects
	 */
	Vector<Rectangle> rectangles;

	/**
	 * Determines if despription is valid, which means parsing of textual despription has been sucessfull
	 */
	boolean valid;


	/**
	 * Constructs description from a textual despription.<br>
	 * <br>
	 * Format:<br>
	 * <code>
	 * filename<br>
	 * image_width image_height<br>
	 * [left top right bottom]; [left top right bottom]; ...
	 * </code>
	 *
	 * @param description textual despription
	 */
	public GroundTruthDescription(String description)
	{
		rectangles = new Vector<Rectangle>();
		parseDescription(description);
	}


	/**
	 * Parses textual despription.<br>
	 * <br>
	 * Format:<br>
	 * <code>
	 * filename<br>
	 * image_width image_height<br>
	 * [left top right bottom]; [left top right bottom]; ...
	 * </code>
	 *
	 * @param description textual despription
	 */
	private void parseDescription(String description)
	{
		//System.out.println("parsing description '"+description+"'");
		
		try
		{
			description = description.trim();
			StringTokenizer st = new StringTokenizer(description, "\n");

			imagePath = st.nextToken();

			imageName = FileUtil.getFileNameWithoutEnding(imagePath);

			String sizeDescription = st.nextToken();
			StringTokenizer stSize = new StringTokenizer(sizeDescription, " ");
			imageSize = new Dimension(Integer.parseInt(stSize.nextToken()), Integer.parseInt(stSize.nextToken()));

			String rectDescription = st.nextToken();
			StringTokenizer stRects = new StringTokenizer(rectDescription, ";");
			while (stRects.hasMoreTokens())
			{
				StringTokenizer stRect = new StringTokenizer(stRects.nextToken(), " ");
				int x1 = Integer.parseInt(stRect.nextToken());
				int y1 = Integer.parseInt(stRect.nextToken());
				int x2 = Integer.parseInt(stRect.nextToken());
				int y2 = Integer.parseInt(stRect.nextToken());
				Rectangle rect = new Rectangle(x1, y1, x2-x1+1, y2-y1+1);

				rectangles.add(rect);
			}

			valid = true;

		}
		catch (Exception e)
		{
			System.err.println("error parsing ground truth description '"+description+"': "+e);
			valid = false;
		}
	}


	/**
	 * Returns true if despription is valid, which means parsing of textual despription has been sucessfull.
	 * False otherwise.
	 */
	public boolean isValid()
	{
		return valid;
	}


	/**
	 * Returns a String containing all values of the ground truth description.
	 */
	@Override
	public String toString()
	{
		String result = this.getClass().getSimpleName()+":\n";
		result += "image name: '"+imageName+"'\n";
		result += "image size: "+imageSize+"\n";
		result += "rectangles: ";
		for (Rectangle rect: rectangles)
		{
			result += rect+"; ";
		}

		return result;
	}


}
