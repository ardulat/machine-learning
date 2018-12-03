import eu.verdelhan.ta4j.TimeSeries;
import eu.verdelhan.ta4j.BaseTick;
import eu.verdelhan.ta4j.BaseTimeSeries;
import eu.verdelhan.ta4j.Tick;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import com.opencsv.CSVReader;
import java.time.ZonedDateTime;
import java.time.ZoneId;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;

/**
 * This class build a Ta4j time series from a CSV file containing ticks.
 */
public class OldCsvLoader {

    private static final DateTimeFormatter DATE_FORMAT = DateTimeFormatter.ofPattern("yyyy-MM-dd");

    /**
     * @return a time series from Dow30 ticks.
     */
    public static TimeSeries loadDow30Series(String filename) {

//        InputStream stream = OldCsvLoader.class.getClassLoader().getResourceAsStream(filename);
        InputStream stream = null;
        System.out.println(filename);
        try {
            stream = new FileInputStream(filename);
        } catch (Exception ex) {
            System.out.println("File not found !");
        }

        List<Tick> ticks = new ArrayList<>();

        CSVReader csvReader = new CSVReader(new InputStreamReader(stream, Charset.forName("UTF-8")), ',', '"', 1);
        try {
            String[] line;
            while ((line = csvReader.readNext()) != null) {
                ZonedDateTime date = LocalDate.parse(line[0], DATE_FORMAT).atStartOfDay(ZoneId.systemDefault());
                double open = Double.parseDouble(line[1]);
                double high = Double.parseDouble(line[2]);
                double low = Double.parseDouble(line[3]);
                double close = Double.parseDouble(line[4]);
                double volume = Double.parseDouble(line[5]);

                ticks.add(new BaseTick(date, open, high, low, close, volume));
            }
        } catch (IOException ioe) {
            Logger.getLogger(OldCsvLoader.class.getName()).log(Level.SEVERE, "Unable to load ticks from CSV", ioe);
        } catch (NumberFormatException nfe) {
            Logger.getLogger(OldCsvLoader.class.getName()).log(Level.SEVERE, "Error while parsing value", nfe);
        }

        return new BaseTimeSeries(filename, ticks);
    }

    public static void main(String[] args) {
        TimeSeries series = OldCsvLoader.loadDow30Series("labeled_data.csv");

        System.out.println("Series: " + series.getName() + " (" + series.getSeriesPeriodDescription() + ")");
        System.out.println("Number of ticks: " + series.getTickCount());
        System.out.println("First tick: \n"
                + "\tVolume: " + series.getTick(0).getVolume() + "\n"
                + "\tOpen price: " + series.getTick(0).getOpenPrice()+ "\n"
                + "\tClose price: " + series.getTick(0).getClosePrice());
    }
}