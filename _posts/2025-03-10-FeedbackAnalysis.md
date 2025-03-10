---
layout: post
title: Uncovering Insights from App Reviews
subtitle: Playing Around with Text Reviews and Ratings
tags: [Unsupervised Learning, NLP, EDA]
comments: true
mathjax: true
categories: [AI, Machine Learning]
author: Nguyen Quoc Anh
date: 2025-03-10
---

I'm trying bag this job that has to do with processing a ton of feedback text data, so I'm playing around with a similar dataset so that I'd get used to the type of data and tech in the field. Here's what I've learned.

I chose this dataset [User Feedback From Top 15 Mobile Apps](https://www.kaggle.com/datasets/mhamidasn/user-feedback-data-from-the-top-15-mobile-apps/data). It has 15,000 datapoints, 1,000 reviews per app with these fields:

- review_id: Unique identifiers for each user feedback/application review.
- content: User-generated feedback/review in text format.
- score: Rating or star given by the user.
- TU_count: Number of likes/thumbs up (TU) received for the review.
- app_id: Unique identifier for each application.
- app_name: Name of the application.
- RC_ver: Version of the app when the review was created (RC).

## Data Exploration
I do some viz! Just to get the feel of the dataset.

<div style="display: flex; flex-direction: column; align-items: center;">
    <!-- First row: 3 square images -->
    <div style="display: flex; gap: 10px;">
        <img src="..\assets\img\feedbacks\score_bars.png" alt="boar" width="200" height="200">
        <img src="..\assets\img\feedbacks\TU_bars.png" alt="monke" width="200" height="200">
        <img src="../assets\img\feedbacks\version_per_app.png" alt="mousedeer" width="200" height="200">
    </div>
    <div style="margin-top: 10px;">
        <img src="../assets\img\feedbacks\tiktok_eda.png" alt="sambar deer" width="620" height="200">
    </div>
    <p style="text-align: center; font-style: italic; margin-top: 10px;">
    Full dataset Statistics: Score Distribution(top-right), Thumb-ups Counts Distribution (top-mid), Word Count Distribution (top-left). Stats for Tiktok (bottom)
    </p>
</div>

As expected, people write reviews when they are completely enraged with the app (as seen with the peak 5 rating). I'm more suprised about the 1500-word review. You couldn't pay me to write that long an essay about an Snapchat's Shrek filter. (jk you can, 🤙 me)

For the purpose of this tiny project, from now on, I will only focus one app, as I doubt the reviews on 1 app would have much to do other reviews on other apps. I picked TikTok, and as all things meme-y and political, it has large counts of extreme 5 and 1 ratings, so much so its score distribution resembles the 🤙 actually.

I also tried to visualize if there was any trends in terms of ratings vs time. Ideally we would see that as time goes on, the users would be more satisfied with the app, more excited about new features. But let's see if that assumption holds

Since we don't have the time uploaded of the review, we'd use sorted version names as a proxy for time. First I plotted the scores against the versions, with the number of thumbs-ups as weights to determine the size of the dot. 

However, I forgot it was discrete scores, so it made this beautiful, but useless plot.

<div style="display: flex; justify-content: center; gap: 10px;">
    <img src="../assets\img\feedbacks\scores_vs_ver.png" alt="distribution" width="100%">
</div>
<p style="text-align: center; font-style: italic; margin-top: 10px;">
Distribution of Scores by Version</p>

So instead, I plotted 3 figures against release versions, and put them on top of each other. The distribution of scores (by percentage of total thumbs-ups counts) per version, mapped against the actual counts of the thumbs ups and reviews. This is much more what I was hoping to see.

<div style="display: flex; justify-content: center; gap: 10px;">
    <img src="..\assets\img\feedbacks\big_plot.png" alt="distribution" width="100%">
</div>
<p style="text-align: center; font-style: italic; margin-top: 10px;">
Percentage of Total Thumbs-ups Counts, per Discrete Scores, per Version (Top), Total Thumbs Up Counts per Versions (Mid), Total Reviews Counts per Version (Bottom) 
</p>


At a glance, we find some really intuitive things:
- The more thumbs-ups/reviews a version got, the more controversial the version was (disagreements of scores among reviews).
- Except for the earlier versions of the app... where there are a lot of disagreements even with high review counts. Uhmmmm interesting, Tiktok, did u hire bots to boost your ratings?

=> You can napkin-test the observation about the thumbsup/review counts vs controversiality actually. Considering the distribution of scores to be a probability distribution, how controversial a version is would be the **entropy** of the ratings. If everyone agrees that a version is, let's say, `3`, then probability distribution would be `[0,0,1,0,0]`, which would make the entropy to be `0`, minimal. But if everyone disagree equally, the prob dist would look something like `[0.2,0.2,0.2,0.2,0.2]`, making the entropy `2.32`. We can plot the entropy, review counts and TU_counts against each other.

<div style="display: flex; justify-content: center; gap: 10px;">
    <img src="../assets\img\feedbacks\line_plot.png" alt="distribution" width="100%">
</div>
<p style="text-align: center; font-style: italic; margin-top: 10px;">
Thumbs Ups Counts vs. Reviews Counts vs Entropy vs Average Scores</p>

We see the entropy spiked almost always when the counts spiked. In fact, the Pearson Correlation between the thumbs-ups/reviews counts with entropy is significant, `0.69` and `0.85`, respectively. So my observation looks good, that the more controversial a version is, the more people will come in and upvote/post reviews.

There is also a trivial correlation between TU_Counts and Review counts being `0.68`, which just implies that people usually agree that the app is probablematic and give feedback at the same time. I should also note that I have no idea how this data was collected, as the curators do not give much info, so there may be a huge [survivorship bias](https://en.wikipedia.org/wiki/Survivorship_bias) in play here.

So what about cases where this is not true, when everyone agrees even though there is low review counts?

| Index | RC_ver  | TU_counts | Review_counts | Entropy | Average score |
|-------|--------|-----------|---------------|---------|---------------|
| 6     | 28.7.3 | 10142     | 17            | 0.74    | 1.28          |
| 9     | 28.8.2 | 78        | 2             | -0.00   | 5.00          |
| 12    | 28.9.3 | 10029     | 18            | 0.45    | 4.78          |
| 18    | 29.1.4 | 13652     | 25            | 0.52    | 3.06          |
| 26    | 29.5.4 | 1098      | 5             | 0.72    | 4.20          |
| 31    | 29.7.3 | 358       | 2             | 0.03    | 1.01          |
| 33    | 29.8.2 | 298       | 2             | 0.29    | 1.10          |
| 37    | 29.9.3 | 2         | 3             | -0.00   | 5.00          |
| 38    | 29.9.4 | 622       | 6             | 0.09    | 4.95          |

I guess if there was only 1 review, the entropy being extremely low makes sense, cuz there was only 1 score. And personally, I don't consider having 2-6 reviews where they all agree anomalies, as they gather mostly under 1000 thumbs up anyway. From now on, I will throw all versions without more than 10 reviews away.



<div style="display: flex; justify-content: center; gap: 10px;">
    <img src="../assets\img\feedbacks\filtered_line_plot.png" alt="distribution" width="100%">
</div>
<p style="text-align: center; font-style: italic; margin-top: 10px;">
Thumbs Ups Counts vs. Reviews Counts vs Entropy vs Average Scores</p>

Seems like after digging into outliers and removing versions without a lot of reviews, there is still no correlation between average scores and review counts, TU_counts or entropy. Without any clue about how the data was collected, we'd have hit a deadend, let's jump into the NLP tasks.

## Topic Clustering
Now, I could do word frequency analysis, word cloud and such, and find which words correlate with lowest scores. But then, I'm sure "bug", "crash" will give low scores and "good", "awesome" will give high scores, and that doesn't really give any interesting insights. Plus, we live in the age of deep learning, and there has got to be so many people working on this already, to develop the state-of-the-arts beyond those fundamental tools.
<!-- 
Furthermore, people could complain about very similar things, yet could give 1 score or 3 score, when these bugs didn't bother them as much.  -->

The [BERTopic](https://maartengr.github.io/BERTopic/) framework has all the bells and whistles I'm hoping for. It's extremely modular and one can  replace its parts with equivalent models, but they all follow the same workflow: First, it will produces embeddings for each document, then apply a dimensionality reduction (UMAP), use a clustering algorithm (HDBSCAN), then infer the word importance for each clusters. This can be achieved using class-based TF-IDF, like normal TF-IDF, but applies on a cluster-level instead of document-level. These will produces keywords most representative of documents in the clusters.

I apply the framework on TikTok's reviews of version 28.9.4, which has the highest entropy of ratings calculated from the section above. For the purpose of demonstrating clustering, high controversiality would guarantee a diversity in opnion, and thus hopefully we'll have some nice clusters!


<table>
    <tr>
        <td colspan="3"><strong>-1_settings_preferences_icons_phone</strong></td>
    </tr>
    <tr>
        <td colspan="3">['settings', 'preferences', 'icons', 'phone', 'annoy', 'annoying', 'text', 'messages', 'inbox', 'language']</td>
    </tr>
    <tr>
        <td style="vertical-align: top;">
            My phone is in English, settings and preferences in app are in English, just why the heck are filters and text to voice translated to Spanish? I see the same filters in English, and I don't know what to do anymore. Please make it that if my preferences are in English, if there is an English version, to switch to the preferred language.
        </td>
        <td style="vertical-align: top;">
            TikTok best and nice app. 100% working, amazing app. This app is a game changer! TikTok has truly revolutionized the way we share and consume content. Whether you're looking for hilarious skits, impressive dance routines, or jaw-dropping talents, this app has it all. The user interface is sleek, intuitive, and easy to navigate. From the moment you open the app, you are greeted with an endless stream of captivating videos that will keep you hooked for hours.
        </td>
        <td style="vertical-align: top;">
            The worst app. The content is amazing 😍. But the experience🤬😡. Your app is terrible. Whenever I open it, I know something will irritate or annoy me. Too many notifications! At least there are some settings for this. Whenever I log in, it tells me about two inbox messages I can never locate. It forever tells me about my contacts following me (no problem with that), but the issue starts when I click on that thing and the only option I have is to follow them back. Stop controlling 🤬.
        </td>
    </tr>
    <tr>
        <td colspan="3"><strong>0_pros_content_platform_youtube</strong></td>
    </tr>
    <tr>
        <td colspan="3">['pros', 'content', 'platform', 'youtube', 'personally', 'experience', 'entertainment', 'ads', 'posts', 'platforms']</td>
    </tr>
    <tr>
        <td style="vertical-align: top;">
            Great app. Too many ads. If you saw a video and lose it... it's lost forever, unfortunately, like so many other apps. I'm starting not to see much diverse videos like when I first joined the app. That was one thing that was enjoyable compared to other platforms. Another con: when going back to look at saved videos that I didn't have time to see originally, there's a lot of deleted content, which I do not like.
        </td>
        <td style="vertical-align: top;">
            Great platform. Amazing algorithm and great user data protection. Easy to use and configure. Very few ads, and the ones I do see are relevant. I use TT to learn about all sorts of things—home repairs, exercises, health tips. And of course, it's entertaining in other ways without being irritating or one-dimensional. The mix of posts is just right, and I like that I can consume videos without having to create them.
        </td>
        <td style="vertical-align: top;">
            Great for growth and meeting people, as well as posting lots of different kinds of content. But the reporting system is BS. I've had posts taken down for responding to hateful people, but people get away with fatphobia, sexism, racism, homophobia, and nudity. The help center needs major improvement. Otherwise, I'm mostly given content I enjoy. I've marked "not interested" on several AI art posts, yet I still see them all the time, which is another flaw I heavily dislike.
        </td>
    </tr>
    <tr>
        <td colspan="3"><strong>1_youtube_bug_slideshow_glitch</strong></td>
    </tr>
    <tr>
        <td colspan="3">['youtube', 'bug', 'slideshow', 'glitch', 'ios', 'recording', 'playlists', 'released', 'redownloading', 'notifications']</td>
    </tr>
    <tr>
        <td style="vertical-align: top;">
            Good app, but there's a glitch that I've had issues with for a while with no solution. There's a bug with saved videos. When I go through my collections of saved videos and try to remove one from the collection, it says it's no longer saved, but it's still there when it shouldn't be. The only way to fix this is to destroy the collection. TikTok needs to fix this bug.
        </td>
        <td style="vertical-align: top;">
            TikTok randomly removed my slideshow feature. I've been trying to get it back by redownloading the app multiple times, but it didn't work. I tried changing settings, still nothing. I have no problems other than that. The app is working fine, it's just the feature that randomly got removed. It seems that some people have this problem too, but not everyone. Please fix this.
        </td>
        <td style="vertical-align: top;">
            It's pretty nice and fun, but since this morning, a lot of people, including me, are complaining about the photo slide feature either suddenly disappearing or, when you post a video using it, it doesn't get any views and sometimes doesn't work. Please fix this bug as soon as possible since some people are genuinely annoyed and upset because of this bug. Thanks!
        </td>
    </tr>
    <tr>
        <td colspan="3"><strong>2_hacked_account_permissions_privacy</strong></td>
    </tr>
    <tr>
        <td colspan="3">['hacked', 'account', 'permissions', 'privacy', 'reinstall', 'settings', 'email', 'access', 'notifications', 'messages']</td>
    </tr>
    <tr>
        <td style="vertical-align: top;">
            Tiktok is sketchy w/ collecting data. You can select no to having access to your contacts, yet it still shows them as "from your contacts". You can limit access from your permissions too, but guess what, still there! Additionally, the updates for having categories for friends versus all FYP is dumb. The notifications changing to just saying someone liked or commented without referencing the comment/post is also super-duper dumb. I'm back on IG because of these changes and privacy break-ins.
        </td>
        <td style="vertical-align: top;">
            Just turn off all Data Sharing options immediately after creating an account from a completely new email created specifically for TikTok. Do the same for that email after creating it. Never connect your contacts or anything else to that account & you can experience TikTok safely & effectively while molding your content the way you want. This way, you avoid being targeted or manipulated more than it's already doing with your emotions 😂.
        </td>
        <td style="vertical-align: top;">
            I was logged out of my business account suddenly on 5/29/23. They switched over information to my personal account, saying it's now a business account AND changed the email, so now my business account doesn't have an email "associated with it". So I can't log in and apparently there's no way to access it. They said it was created this way, but how could I have done that without an email/tel# etc?! I have PROOF that I own both accounts, it was a TikTok error, and now no one's answering or helping me.
        </td>
    </tr>
    <tr>
        <td colspan="3"><strong>3_android_glitching_screen_phone</strong></td>
    </tr>
    <tr>
        <td colspan="3">['android', 'glitching', 'screen', 'phone', 'filters', 'swipe', 'phones', 'tablet', 'try', 'stopped']</td>
    </tr>
    <tr>
        <td style="vertical-align: top;">
            The app used to be entertaining, now it's just angering. Can't try filters in peace without a message popping up in the middle of the screen: "Microphone is off when a sound is added." Okay, one time is fine, but EVERY TIME I want to try a different filter? Sigh. The photo posts where you can't even pause because it shows the caption in a weird format—like, no one asked for that! We were fine reading long captions the way they were before. So many more things, but I'm running out of characters.
        </td>
        <td style="vertical-align: top;">
            Would be a 5 if things worked more smoothly. Switched to an Android and found it annoying to try to like comments because the hitbox was so small and it always thinks I'm trying to reply. Also, I wish you could just swipe out of videos instead of reaching up to the back button, but that one's more niche of a problem. Things are always glitching, but I'm so addicted to the app that I look past it.
        </td>
        <td style="vertical-align: top;">
            THIS IS THE WORST APP EVER! If I could rate it zero stars, I WOULD! It's glitching when I'm recording a video, I can't use the beautify filters in peace BECAUSE IT'S NOT WORKING. When I exit the app, I can still hear the videos LOUDLY in the background. It's not letting me scroll to see other videos, and when I post a private video, it makes it public. I can't follow anymore because it's not letting me, AND IT'S GETTING ON MY NERVES! I have a lot more to share, but I'm running out of characters.
        </td>
    </tr>
    <tr>
        <td colspan="3"><strong>4_issues_issue_loading_wifi</strong></td>
    </tr>
    <tr>
        <td colspan="3">['issues', 'issue', 'loading', 'wifi', 'network', 'fix', 'trouble', 'broken', 'fixed', 'unstable']</td>
    </tr>
    <tr>
        <td style="vertical-align: top;">
            Request! It's a really great app! I've been using this app for years on different devices, but the problem is that when we make a video, it takes hours to see the complete video. It always gets blocked after a few minutes, and I'm unable to see the video. While making videos, it glitches every time, which is very annoying. I want you to fix this, or else other users may face the same situation. Other than that, it's a great app for entertainment purposes!
        </td>
        <td style="vertical-align: top;">
            Will update my review if these issues get fixed, but I'm super frustrated. Network connection error messages constantly appear on WiFi or data (I work from home, and my internet is great). Duets from drafts are broken (get next to no reach at all). Almost every video gets around 200-300 views and then falls off—it doesn't seem to matter the quality or content (something many creators, even those with many followers, have been noticing in the last few months).
        </td>
        <td style="vertical-align: top;">
            Request: I will update my review if these issues get fixed. But I'm super frustrated. Network connection error messages constantly appear on WiFi or data (I work from home, and my internet is great). Duets from drafts are broken (get next to no reach at all). Almost every video gets around 200-300 views and then falls off—it doesn't seem to matter the quality or content. Many creators, even those with many followers, have been noticing this issue in the last few months.
        </td>
    </tr>
    <tr>
        <td colspan="3"><strong>5_followers_unfollows_profile_account</strong></td>
    </tr>
    <tr>
        <td colspan="3">['followers', 'unfollows', 'profile', 'account', 'follow', 'accounts', 'issue', 'reinstalling', 'followed', 'glitches']</td>
    </tr>
    <tr>
        <td style="vertical-align: top;">
            New account. Can't set up a name, can't set up a profile picture, can't even follow anyone—it just refreshes back to the follow button. So far, a horrible experience, and I've been here for less than 24 hours.
        </td>
        <td style="vertical-align: top;">
            There have been three glitches that I keep coming across. 1: The favorite button—every time I tried to add something to favorites, it wouldn’t work. I tried everything: uninstalling and reinstalling, restarting my phone over and over again, logging in and out, and more. 2 and 3 are the same issue but with the like button and follow button. It's so annoying and frustrating that it makes the app nearly impossible to use.
        </td>
        <td style="vertical-align: top;">
            Unusable. I downloaded this app, made an account, and put my birthdate (24 years old), and it automatically made my account private with no option to change it to public. Then, when we finally got the app to make a correct account, it wouldn't let me follow anyone. The second I follow someone, it immediately unfollows them.
        </td>
    </tr>
</table>

With the result being clustered into 6 clusters, we can use the keywords and most representative docs to get a sense of what these clusters represent.
* `-1` is outliers, so you can see their content diverges a lot from each others.
* `0` you can see a lot mentions of ads and the content of the Tiktok shorts itselves.
* `1` is more about technical bugs, especially about the slideshow feature. If you sort the reviews of this cluster by their Thumbs Up, these are also the most upvoted issues!
* `2` is more about permissions and privacy settings and concerns.
* `3` is various concerns about the UX. These are the most varied cluster just from vibes.
* `4` is about issues with network connections
* `5` is about follow and favorite buttons and setting up users counts.

I'll admit, this clustering-into-topic algorithm is more useful than I thought. Intuitively, when people visit the app's page on the app store, mostly to complain or (they're paid to) praise, if they see another person already making the same complaint, they would just upvote and not submit their own reviews. Which is to say, each problem would be summarized by 1 review, with a lot of people voting and nodding their heads, and it wouldn't make sense to do clustering. However, it seems that people would discover different bugs about the same features, which then, since their contents are similar, can be neatly classified into clusters. For practical purposes, instead of having to read all hundreds of reviews, or ONLY reading the most upvoted ones and potentially miss critical feedbacks, these tools can be used to categorize reviews! The curator would be able to pass each clusters into the team responsible to the feature (UX to design team, ads to recommendation team).

"neat" here is not perfect of course, as you can see by ranking them by upvotes within a cluster, there are a lot of variations between the top reviews. And of course, the most unique keywords by TF-IDF is flawed, since they relied on word frequency and not semantics to determine representitiveness. Given more time and compute, one can definitely improve the results. One can use the hierachy extracted to consider merging the clusters when it makes sense. The thumbs up counts present an informative parameter, which can be integrated and used to improve various modules in Bertopic. For example, creating weighted sub-graphs between points in the earlier stage of DBSCAN can incorporate information about upvotes, and not just use cosine similarity. One can also pass these most representitive docs through an LLM for it to give better summary and list of keywords. While we're at the topic of LLM, you can ask the LLM to rewrite the reviews in specific format, like {user, feature, issue, keyword}, which would help Bertopic even more. Hell, with Gemini's 1M context windows, you should be able to put all 150 reviews or so of 1 version. Its performance? You'd have to try it and tell me. Needle-in-haystack evaluation is kinda wack for how silly it is, but that's just my humble opinion. Idk if it is the best representation of long-context capabilities.

That leads us to elephant in the room, which is evaluation. In all cases, all applications need to evaluated to understand its performance and how to improve it. However, this dataset does not give classification information, and thus there's no way for us quantitatively test our clustering. One can argue that LLM is coming close to human intelligence and use it to create pseudo-labels for this task, but, again, that assumption itself would require testing. As for me, I just want to explore a dataset and try out tech for a simple application. This is too much excitement for one day already, so Imma leave the rest of the work as homework for my dear reader(s). All my notebooks and visualizations can be found [here](https://github.com/BatmanofZuhandArrgh/feedback_analysis). Peace! 