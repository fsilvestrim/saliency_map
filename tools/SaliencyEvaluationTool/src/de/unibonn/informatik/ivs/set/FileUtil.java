
package de.unibonn.informatik.ivs.set;


import java.util.*;
import java.io.*;
import java.nio.channels.*;


/**
 * Provides supporting methods for handling files
 *
 * @author		Bernd Wendt
 * @version		2011.0314
 *
 */
public class FileUtil
{

	/**
	 * Copies files of folder <code>pathToCopyFrom</code> (incl. subfolders) to folder <code>pathToCopyTo</code>
	 * if and only if in folder <code>pathDefiningNames</code> there exists a file with the same name.
	 * File endings may be different.
	 *
	 * Note: If <code>pathToCopyFrom</code> contains different files with the same name, any of these files will be copied.
	 *
	 * @param	pathToCopyFrom	folder to copy from
	 * @param	pathToCopyTo folder to copy to
	 * @param	pathDefiningNames folder containing files that define the names of files to copy
	 */
	public static void copyDefinedFiles(String pathToCopyFrom, String pathToCopyTo, String pathDefiningNames)
	{

	 // retrieve filenames to copy
	 File dirDefiningNames = new File(pathDefiningNames);
	 if (!dirDefiningNames.exists())
	 {
		 System.err.println("directory '"+dirDefiningNames+"' does not exist!");
		 return;
	 }
	 if (!dirDefiningNames.isDirectory())
	 {
		 System.err.println("file '"+dirDefiningNames+"' is not a directory!");
		 return;
	 }

	 File[] files = dirDefiningNames.listFiles();
	 Vector<String> fileNames = new Vector<String>();
	 for (int iFile=0; iFile<files.length; iFile++)
	 {
		 fileNames.add(getFileNameWithoutEnding(files[iFile].getName()));
	 }


	 // retrieve files in directory to copy from
	 File dirToCopyFrom = new File(pathToCopyFrom);
	 if (!dirToCopyFrom.exists())
	 {
		 System.err.println("directory '"+pathToCopyFrom+"' does not exist!");
		 return;
	 }
	 if (!dirToCopyFrom.isDirectory())
	 {
		 System.err.println("file '"+pathToCopyFrom+"' is not a directory!");
		 return;
	 }
	 Map<String, File> filesInCopyDir = new TreeMap<String, File>();
	 addFilesToMap(filesInCopyDir, dirToCopyFrom);


	 // copy files to destination
	 File dirDestination = new File(pathToCopyTo);
	 if (!dirDestination.exists())
	 {
		 dirDestination.mkdirs();
	 }
	 else if (!dirToCopyFrom.isDirectory())
	 {
		 System.err.println("file '"+pathToCopyTo+"' is not a directory!");
		 return;
	 }
	 for (String fileName:fileNames)
	 {
		 File foundFile = filesInCopyDir.get(fileName);

		 if (foundFile == null) System.err.println("could not find file '"+fileName+"' in folder '"+pathToCopyFrom+"' and its subfolders");
		 else copyFile(foundFile, new File(dirDestination.getAbsolutePath()+File.separator+foundFile.getName()));
	 }

	}


	/**
	 * Adds all files included in a folder and its subfolders to a map.
	 * Keys for the map are file names without endings.
	 *
	 * @param map	map to add files to
	 * @param file	directory including files
	 */
	public static void addFilesToMap(Map<String, File> map, File file)
	{
		if (!file.exists()) return;

		if (file.isDirectory())
		{
			File[] children = file.listFiles();
			for (int iFile=0; iFile<children.length; iFile++)
			{
				addFilesToMap(map, children[iFile]);
			}
		}
		else
		{
			map.put(getFileNameWithoutEnding(file.getName()), file);
		}
	}


	/**
	 * Reads a file and stores its contents in a String.
	 * CR+LF and CR are converted to LF.
	 * 
	 * @param in	The file to read
	 * @return		The file's contents
	 */
	public static String readFile(File in)
	{
		String result = null;

		try
		{
			StringBuilder builder = new StringBuilder();
			String line;

			BufferedReader reader = new BufferedReader(new FileReader(in));

			while ((line = reader.readLine()) != null)
			{
				builder.append(line);
				builder.append('\n');
			}

			reader.close();

			result = builder.toString();
		}
		catch (IOException e)
		{
			System.err.println("error reading file '"+in.getAbsolutePath()+"': "+e);
		}

		return result;
	}


	/**
	 * Copies a file.
	 *
	 * @param in	File to copy
	 * @param out	File to copy to
	 */
	public static void copyFile(File in, File out)
	{
		FileChannel inChannel = null;
		FileChannel outChannel = null;

		try
		{
			inChannel = new FileInputStream(in).getChannel();
			outChannel = new FileOutputStream(out).getChannel();

			inChannel.transferTo(0, inChannel.size(),	outChannel);
		}
		catch (IOException e)
		{
			System.err.println("could not copy file '"+in.getAbsolutePath()+"' to '"+out.getAbsolutePath()+"': "+e.toString());
		}
		finally
		{
			if (inChannel != null) try { inChannel.close(); } catch (IOException e1) {}
			if (outChannel != null) try { outChannel.close(); } catch (IOException e1) {}
		}

	}

	
	/**
	 * Determines the file name without ending for a file path.
	 *
	 * @param	filePath	file path with or without ending
	 * @return				file name without ending
	 */
	public static String getFileNameWithoutEnding(String filePath)
	{
		int pos1 = filePath.lastIndexOf("\\")+1;
		if (pos1 == 0) pos1 = filePath.lastIndexOf("/")+1;
		if (pos1 == 0) pos1 = 0;

		int pos2 = filePath.lastIndexOf(".");
		if (pos2 == -1) pos2 = filePath.length();

		return filePath.substring(pos1, pos2);
	}


	/**
	 * Searches in a set of files for one whose name contains a specified name.<br>
	 * If more than one file in the set contain the name, it is unspecified which one is returned.
	 *
	 * @param fileName name to search for
	 * @param files    array of files to serach in
	 *
	 * @return         file whose name contains the name searched for
	 */
	public static File findFile(String fileName, File[] files)
	{
		for (File file: files)
		{
			if (file.getName().contains(fileName)) return file;
		}

		return null;
	}


}
