import pandas as pd


if __name__ == "__main__":
    seasons = pd.read_csv("data/seasons.csv")
    season_start = seasons.set_index("seasonId")["seasonStartDate"].to_dict()
    season_end = seasons.set_index("seasonId")["seasonEndDate"].to_dict()
    date_df = []
    for year in range(2018, 2022):
        year_data = seasons.loc[seasons.seasonId == year]
        dates = pd.DataFrame({"date": pd.date_range(f"{year}-01-01", f"{year}-12-31")})
        dates["seasonflag"] = 0
        dates.loc[
            dates.date.between(
                year_data.preSeasonStartDate.iloc[0], year_data.preSeasonEndDate.iloc[0]
            ),
            "seasonflag",
        ] = 1
        dates.loc[
            dates.date.between(year_data.seasonStartDate.iloc[0], year_data.seasonEndDate.iloc[0]),
            "seasonflag",
        ] = 2
        dates.loc[
            dates.date.between(
                year_data.postSeasonStartDate.iloc[0], year_data.postSeasonEndDate.iloc[0]
            ),
            "seasonflag",
        ] = 3
        dates.loc[dates.date == year_data.allStarDate.iloc[0], "seasonflag"] = 4

        dates["season_start"] = year_data.seasonStartDate.iloc[0]
        dates["season_end"] = year_data.seasonEndDate.iloc[0]
        dates["all_star"] = year_data.allStarDate.iloc[0]
        date_df.append(dates)
    date_df = pd.concat(date_df)
    date_df["date"] = date_df.date.apply(lambda x: x.strftime("%Y%m%d"))
    date_df.to_csv("data/seasons_formatted.csv", index=False)
