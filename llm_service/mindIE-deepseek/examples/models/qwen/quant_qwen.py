# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import argparse
import sys
import logging
import torch
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlierConfig, AntiOutlier
from atb_llm.models.base.model_utils import safe_get_tokenizer_from_pretrained
from atb_llm.models.base.model_utils import safe_get_model_from_pretrained

#boolq的校准数据
data_list_1 = [
        "Ghost in the Shell -- Animation studio Production I.G has produced several \
        different anime adaptations of Ghost in the Shell, starting with the 1995 film \
        of the same name, telling the story of Section 9's investigation of the Puppet \
        Master. The television series Ghost in the Shell: Stand Alone Complex followed \
        in 2002, telling an alternate story from the manga and first film, featuring \
        Section 9's investigations of government corruption in the Laughing Man and \
        Individual Eleven incidents. A sequel to the 1995 film, Ghost in the Shell 2: \
        Innocence, was released in 2004. In 2006, the film Ghost in the Shell: Stand \
        Alone Complex - Solid State Society retook the story of the television series. \
        2013 saw the start of the Ghost in the Shell: Arise original video animation \
        (OVA) series, consisting of four parts through mid-2014. The series was \
        recompiled in early 2015 as a television series titled Ghost in the Shell: \
        Arise - Alternative Architecture, airing with an additional two episodes (one \
        part). An animated feature film produced by most of the Arise staff, titled \
        Ghost in the Shell: The New Movie, was released on June 20, 2015. A live-action \
        American film of the same name was released on March 31, 2017.\nQuestion: is \
        ghost in the shell based on the anime?\nAnswer:",
        "Porpoise -- Porpoises are a group of fully aquatic marine mammals that are \
        sometimes referred to as mereswine, all of which are classified under the \
        family Phocoenidae, parvorder Odontoceti (toothed whales). There are seven \
        extant species of porpoise. They are small toothed whales that are very closely \
        related to oceanic dolphins. The most obvious visible difference between the \
        two groups is that porpoises have shorter beaks and flattened, spade-shaped \
        teeth distinct from the conical teeth of dolphins. Porpoises, and other \
        cetaceans, belong to the clade Cetartiodactyla with even-toed ungulates, \
        and their closest living relatives are the \
        hippopotamuses, having diverged from them about 40 million years \
        ago.\nis a porpoise and a dolphin the same animal?\nAnswer:",
        "Onyx -- Brazilian green onyx was often used as plinths for art deco sculptures \
        created in the 1920s and 1930s. The German sculptor Ferdinand Preiss used \
        Brazilian green onyx for the base on the majority of his chryselephantine \
        sculptures. Green onyx was also used for trays and pin dishes -- produced \
        mainly in Austria -- often with small bronze animals or figures attached.\n\
        Question: is there such a thing as green onyx?\nAnswer:",
        "Wachovia -- The acquisition of Wachovia by Wells Fargo was completed on \
        December 31, 2008 after a government-forced sale to avoid Wachovia's failure. \
        The Wachovia brand was absorbed into the Wells Fargo brand in a process that \
        lasted three years: on October 15, 2011, the last Wachovia branches in North \
        Carolina were converted to Wells Fargo.\nQuestion: is wells fargo and wachovia \
        the same bank?\nAnswer:",
        "Friday Night Lights (film) -- Friday Night Lights is a 2004 American sports \
        drama film, directed by Peter Berg, which 'dramatized' the coach and players \
        of a high school football team in the Texas city of Odessa that supported and \
        was obsessed with them. The book on which it was based, Friday Night Lights: \
        A Town, a Team, and a Dream (1990) by H.G. Bissinger, followed the story of \
        the 1988 Permian High School Panthers football team as they made a run towards \
        the state championship. A television series of the same name premiered on \
        October 3, 2006 on NBC. The film won the Best Sports Movie ESPY Award and was \
        ranked number 37 on Entertainment Weekly's list of the Best High School Movies.\
        \nQuestion: is friday night lights movie based on a true story?\nAnswer:",
        "Peace bond -- The use of peace bonds is rather uncommon in the U.S. justice \
        system, but a deferred prosecution has a similar effect. Since there is no \
        conviction or admission of any guilt, signing a peace bond in Canada does not \
        usually result in U.S. inadmissibility under INA \u00a7 212 (a) (2).\nQuestion: \
        is a peace bond an admission of guilt?\nAnswer:",
        "Eating mucus -- Mucophagy, despite its benefits on one's immunity, comes with \
        some health risks due to the potential physical aggravation resulting from the \
        action of nose picking, and the germs on fingers and in mucus. Picking one's \
        nose can cause upper airway irritation as well as other injuries including \
        nasal septal perforation (a ``through-and-through defect'' of the cartilage \
        separating the nostrils), and epistaxis (nosebleed). In a study by Andrade and \
        Srihari, 25% of subjects were ailed by nose bleeds, 17% with nasal infections, \
        and 2% with damage more serious than bleeding. W. Buzina studied the fungal \
        diversity in nasal mucus in 2003. 104 samples were gathered with 331 identifiable \
        strains of fungi and 9 different species per patient.\nQuestion: does eating your \
        boogers improve your immune system?\nAnswer:",
        "Venomoid -- A venomoid is a venomous snake that has undergone a surgical \
        procedure to remove or inhibit the production of snake venom. This procedure \
        has been used for venomous snakes kept as pets or used in public demonstrations \
        in order to remove the risk of injury or death when handled. The removal of \
        venom glands or fangs of exhibited animals may be by surgery or simple mutilation; \
        some or all of these procedures have been considered illegal and unethical. \
        Removal of fangs is uncommon, as snakes frequently regenerate teeth, and the \
        more invasive procedure of removing the underlying maxillary bone would be fatal. \
        Most venomoid procedures consist of either removing the venom gland itself, or \
        severing the duct between the gland and the fang. However, the duct and gland \
        have been known to regenerate, and supposedly ``safe'' snakes have killed mice \
        and successfully envenomated humans.\nQuestion: can you remove the venom glands \
        from a snake?\nAnswer:",
        "Big Boss (Metal Gear) -- Big Boss is one of the central characters in the Metal \
        Gear video game series. He was introduced in the original Metal Gear games for \
        the MSX2 as the commanding officer and subsequent nemesis of Solid Snake. He is \
        later featured as Naked Snake, the protagonist of Metal Gear Solid prequels \
        where he is initially depicted as an American Special Forces Operator and \
        decorated war hero until political manipulations cause him to be disillusioned \
        and start his own private mercenary company. Big Boss's character has been \
        praised by video game publications for his role as a villain as well for his \
        relationship with Solid Snake. As the series' chronology progressed, his exact \
        allegiance and motivations became increasingly complex; his first appearances \
        are depicted as a traitor dreaming of a world of perpetual war, but subsequent \
        appearances have revealed him to be a key figure in an ideological dispute that \
        shaped the latter half of the twentieth century and a man whose conscience was \
        disturbed by the attitude of leaders towards soldiers, prompting his decision to \
        become a soldier of fortune and Venom Snake's mental template.\nQuestion: is \
        solid snake and big boss the same person?\nAnswer:",
        "Jessie (2011 TV series) -- After casting was finalized and changes were made to \
        several of the characters to suit the actors chosen, the series skipped the pilot \
        phase and was put directly into production. Filming began in June 2011 on Stage \
        3/8 at Hollywood Center Studios which, prior to start of production, served as \
        the sound stage where the Disney Channel series Wizards of Waverly Place was \
        taped. 13 episodes were originally ordered for the first season, but while the \
        show's first season was in production, Disney Channel ordered an additional seven \
        episodes, bringing the total number of episodes for the first season to 20. When \
        asked about the atmosphere on set during an interview with MSN TV, Ryan described \
        her relationship with the young cast: ``I definitely feel like a nanny! They are \
        smart kids, but they're real kids. They like to have fun. My policy is: We can \
        play hard, as long as we work hard, and because we work hard, we need to play \
        hard.'' Filming on the series wrapped on February 22, 2015.\nQuestion: is the \
        show jessie filmed in new york?\nAnswer:",
        "Song of Songs -- The Song of Songs, also Song of Solomon or Canticles \
        (Hebrew: \u05e9\u05b4\u05c1\u05d9\u05e8 \u05d4\u05b7\u05e9\u05b4\u05bc\u05c1\
        \u05d9\u05e8\u05b4\u05d9\u05dd\u202c, \u0160\u00eer Ha\u0161\u0160\u00eer\
        \u00eem, Greek: \u1f8e\u03c3\u03bc\u03b1 \u1f8e\u03c3\u03bc\u03ac\u03c4\u03c9\
        \u03bd, asma asmaton, both meaning Song of Songs), is one of the megillot \
        (scrolls) found in the last section of the Tanakh, known as the Ketuvim (or \
        ``Writings''), and a book of the Old Testament.\nQuestion: is the song of songs \
        the same as the song of solomon?\nAnswer:",
        "Northwest Florida State College -- The school voted to change its name to \
        Okaloosa-Walton Community College in 1988, and gained four-year status in 2003, \
        thus changing its name to Okaloosa-Walton College.\nQuestion: is northwest florida \
        state college a 4 year college?\nAnswer:",
        "A Quiet Place (film) -- A Quiet Place is a production of Sunday Night and \
        Platinum Dunes; it was produced on a budget of $17 million. Krasinski wrote the \
        screenplay with story co-writers Scott Beck and Bryan Woods. Beck and Woods grew \
        up together in the US state of Iowa, and had watched numerous silent films in \
        college. By 2013, they began working on the story that would lead to the film. \
        They used their experience growing up close to farmland as the basis, including \
        a grain silo setting as a place considered dangerous in their upbringing. They \
        initiated their approach with a 15-page proof of concept. Initially, the writers \
        had considered developing the film into a Cloverfield installment, but after \
        pitching their ideas to the studio collectively, all of those involved decided \
        to keep the film its own entity.\nQuestion: is the movie the quiet place based \
        on a book?\nAnswer:",
        "2018 FIFA World Cup qualification \u2013 UEFA Group G -- The group winners, \
        Spain, qualified directly for the 2018 FIFA World Cup. The group runners-up, \
        Italy, advanced to the play-offs as one of the best 8 runners-up, where they \
        lost to Sweden and thus failed to qualify for the first time since 1958.\n\
        Question: did spain qualify for the 2018 world cup?\nAnswer:",
        "Red squirrel -- The eastern grey squirrel and the red squirrel are not directly \
        antagonistic, and violent conflict between these species is not a factor in the \
        decline in red squirrel populations. However, the eastern grey squirrel appears \
        to be able to decrease the red squirrel population due to several reasons:\n\
        Question: are grey and red squirrels the same species?\nAnswer:",
        "Bermuda -- Bermuda is a group of low-forming volcanoes in the Atlantic Ocean, \
        near the western edge of the Sargasso Sea, roughly 578 nautical miles (1,070 km; \
        665 mi) east-southeast of Cape Hatteras on the Outer Banks of North Carolina and \
        about 594 nautical miles (1,100 km; 684 mi) southeast of Martha's Vineyard of \
        Massachusetts. It is 898 nautical miles (1,663 km; 1,033 mi) northeast of Miami, \
        Florida, and 667 nautical miles (1,235 km; 768 mi) from Cape Sable Island, in \
        Nova Scotia, Canada. The islands lie due east of Fripp Island, South Carolina, \
        west-northwest of Cape Verde, southeast of New York City, New York, north-northwest \
        of Brazil and 1,759 km (1,093 mi) north of Cuba.\nQuestion: is bermuda off the \
        coast of south carolina?\nAnswer:",
        "The People's Court -- The losing party does not actually need to pay the judgment, \
        as such. Instead (as is stated in the disclaimer at the end of each show), both \
        parties are paid from a fund (set up by Ralph Edwards-Stu Billett Productions). \
        This fund was based on the amount of the lawsuit claim, but an exact formula was \
        not stated. The fund was to be first divided equally, then any monetary judgment \
        ordered was subtracted from the loser's half (and presumably both halves in the \
        case of cross judgments). Each litigant received at least what remained of their \
        half in shows concluding with that disclaimer.\nQuestion: do litigants on people's \
        court get paid?\nAnswer:",
        "Texas -- Texas (/\u02c8t\u025bks\u0259s/, locally /-s\u0259z/; Spanish: Texas or \
        Tejas (\u02c8texas)) is the second largest state in the United States by both area \
        and population. Geographically located in the South Central region of the country, \
        Texas shares borders with the U.S. states of Louisiana to the east, Arkansas to the \
        northeast, Oklahoma to the north, New Mexico to the west, and the Mexican states of \
        Chihuahua, Coahuila, Nuevo Le\u00f3n, and Tamaulipas to the southwest, while the \
        Gulf of Mexico is to the southeast.\nQuestion: is texas the biggest state in the \
        us?\nAnswer:",
        "The Adventures of Tintin (film) -- Spielberg acquired rights to produce a film \
        based on The Adventures of Tintin series following Herg\u00e9's death in 1983, and \
        re-optioned them in 2002. Filming was due to begin in October 2008 for a 2010 \
        release, but release was delayed to 2011 after Universal opted out of producing \
        the film with Paramount, who provided $30 million on pre-production. Sony chose \
        to co-produce the film. The delay resulted in Thomas Sangster, who had been \
        originally cast as Tintin, departing from the project. Producer Peter Jackson, \
        whose company Weta Digital provided the computer animation, intends to direct a \
        sequel. Spielberg and Jackson also hope to co-direct a third film. The world \
        premi\u00e8re took place on 22 October 2011 in Brussels. The film was released in \
        the United Kingdom and other European countries on 26 October 2011, and in the \
        United States on 21 December 2011, in Digital 3D and IMAX.\nQuestion: will there \
        be a adventures of tintin 2?\nAnswer:",
        "Emma Pillsbury -- Emma Pillsbury Schuester (previously Pillsbury-Howell) is a \
        fictional character from the Fox musical comedy-drama series Glee. Portrayed by \
        actress Jayma Mays, Emma has appeared in Glee from its pilot episode, first \
        broadcast on May 19, 2009. Emma was developed by Glee creators Ryan Murphy, Brad \
        Falchuk and Ian Brennan. She is a guidance counselor at the fictional William \
        McKinley High School in Lima, Ohio where the series is set. Emma suffers from \
        obsessive-compulsive disorder and has romantic feelings for glee club director \
        Will Schuester (Matthew Morrison), but becomes engaged to football coach Ken \
        Tanaka (Patrick Gallagher) as Will is married. Ken ultimately breaks up with her \
        on their wedding day because of her feelings for Will, and when Will leaves his \
        wife Terri (Jessalyn Gilsig), he and Emma share a kiss. Their relationship is \
        short-lived, and in the second season, Emma and her dentist boyfriend Carl Howell \
        (John Stamos) marry in Las Vegas. The wedding is later annulled as it was \
        unconsummated. At the beginning of the third season, she and Will are living \
        together; they become engaged shortly after New Years, and consummate their \
        relationship near the end of the school year. Emma leaves Will at the altar \
        midway through the fourth season, but the two later reconcile and marry in the \
        season finale. She becomes pregnant during the middle of the fifth season.\n\
        Question: do will and emma get together in glee?\nAnswer:",
        "The Princess and the Goblin (film) -- The Princess and the Goblin (Hungarian: \
        A hercegn\u0151 \u00e9s a kobold) is a 1991 British-Hungarian-American animated \
        musical fantasy film directed by J\u00f3zsef G\u00e9mes and written by Robin \
        Lyons, an adaptation of George MacDonald's 1872 novel of the same name.\n\
        Question: is the princess and the goblin a disney movie?\nAnswer:",
        "WWE draft -- On May 25, 2016, due to SmackDown moving to Tuesdays and to a live \
        broadcast starting July 19, necessitating a brand extension, WWE announced that \
        the draft would be returning. It would later be announced that the 2016 WWE draft \
        would take place on July 19 during SmackDown's first live broadcast, which was \
        also the first time that the draft took place on SmackDown. The 2017 draft was \
        labeled the Superstar Shake-up as instead of a traditional draft, the general \
        managers of Raw and SmackDown could trade and make deals between their respective \
        talent.\nQuestion: is there going to be a wwe draft in 2017?\nAnswer:",
        "Izzie Stevens -- Heigl garnered critical acclaim for her performance as Izzie \
        and received numerous awards and nominations for her role, winning the \
        ``Outstanding Supporting Actress In A Drama Series'' at the 2007 Emmy Awards. \
        She was critical of the character's development during the show's fourth season, \
        particularly her romance with George. She declined to put herself forward for the \
        2008 Emmy Awards, citing insufficient material in the role. After speculation \
        that Izzie would be killed off in the fifth season, the character was diagnosed \
        with Stage 4 metastatic melanoma. She married Alex in the series' one-hundredth \
        episode, and afterwards, her tumor was successfully removed. Izzie made her final \
        appearance in the sixth season, leaving Seattle after Alex refused to resume their \
        marriage. Heigl requested to be released from her contract 18 months early, in \
        order to spend more time with her family. In January 2012, Heigl reported that \
        she would like to return to Grey's Anatomy to give closure to her character, \
        however, Rhimes confirmed that there were no plans to have the character return \
        at that time and has since stated that she has no plans to ever re-approach \
        Izzie's storyline again.\nQuestion: does izzie come back in grey's anatomy?\n\
        Answer:",
        "Sam Beckett -- When Sam corrected the timeline, he leaped forward, but not all \
        the way home; this time, he found himself assuming the identity of a minor-league \
        professional baseball player named Tim Fox. For the rest of his life (an epilogue \
        in the series finale tells us Sam never gets home, but in our terms, it was the \
        next four years/five seasons, the duration of the show) Sam would continue to \
        travel back and forth through time; swapping identities with various people and \
        as a tagline for the show reiterated, ``setting right what once went wrong.''\n\
        Question: did sam ever make it home in quantum leap?\nAnswer:",
        "Safety (gridiron football score) -- In gridiron football, the safety (American \
        football) or safety touch (Canadian football) is a scoring play that results in \
        two points (or, in rare cases, one point) being awarded to the scoring team. \
        Safeties can be scored in a number of ways, such as when a ball carrier is \
        tackled in his own end zone or when a foul is committed by the offense in their \
        own end zone. After a safety is scored in American football, the ball is kicked \
        off to the team that scored the safety from the 20-yard line; in Canadian \
        football, the scoring team also has the options of taking control of the ball at \
        their own 35-yard line or kicking off the ball, also at their own 35-yard line. \
        The ability of the scoring team to receive the ball through a kickoff differs \
        from the touchdown and field goal, which require the scoring team to kick the \
        ball off to the scored upon team. Despite being of relatively low point value, \
        safeties can have a significant impact on the result of games, and Brian Burke \
        of Advanced NFL Stats estimated that safeties have a greater abstract value than \
        field goals, despite being worth a point less, due to the field position and \
        reclaimed possession gained off the safety kick.\nQuestion: is it possible to get \
        1 point in football?\nAnswer:",
        "Atomic number -- The atomic number or proton number (symbol Z) of a chemical \
        element is the number of protons found in the nucleus of an atom. It is identical \
        to the charge number of the nucleus. The atomic number uniquely identifies a \
        chemical element. In an uncharged atom, the atomic number is also equal to the \
        number of electrons.\nQuestion: is the atomic number equal to the number of \
        protons?\nAnswer:"
    ]

#ceval的校准数据
data_list_2 = [
        ["下列关于税法基本原则的表述中，不正确的是____。\nA. 税收法定原则包括税收要件法定原则和税务合法性原则\nB. 税收公平原则源于法律上的平等\
        性原则\nC. 税收效率原则包含经济效率和行政效率两个方面\nD. 税务机关按法定程序依法征税，可以自由做出减征、停征或免征税款的决定\nAnswer:"],
        ["求极限：$\lim_{x\rightarrow0}\frac{\int_{x^2}^x{\frac{\sin\left(xt\right)}{t}}\mathrm{d}t}{x^2}=$____,\nA. \
        $\frac{5}{6}$\nB. 1\nC. $\frac{7}{6}$\nD. $\frac{4}{3}$\nAnswer:"],
        ["蓝印花布是一种传统的民间纺织印染工艺品。蓝印花布印制方法始于____。\nA. 汉代\nB. 魏晋时期\nC. 唐代\nD. 宋代\nAnswer:"],
        ["不属于第二信使的物质是____\nA. cAMP\nB. cGMP\nC. $IP_3$\nD. 胰岛素\nAnswer:"],
        ["最早对管理的具体职能加以概括和系统论述的是____。\nA. 泰罗\nB. 法约尔\nC. 孔茨\nD. 韦伯\nAnswer:"],
        ["英语名词“lab”(实验室)原来的形式是“laboratory”，这在词的形成方式上属于____。\nA. 直接成词\nB. 变形成词\nC. 变性成词\n\
        D. 逆序成词\nAnswer:"],
        ["甲与乙准备进行一个游戏：向空中扔三枚硬币，如果它们落地后全是正面向上或全是反面向上，乙就给甲钱；但若出现两正面一反面或两反面一正\
        面的情况，则由甲给乙钱。乙要求甲每次给10元，那么，从长远来看，甲应该要求乙每次至少给____元才可考虑参加这个游戏。\nA. 10\n\
        B. 15\nC. 20\n\
        D. 30\nAnswer:"],
        ["治疗糖尿病酮症酸中毒时最应注意的电解质紊乱是____\nA. 低钠血症\nB. 低钾血症\nC. 高氯血症\nD. 高钙血症\nAnswer:"],
        ["估计下列分子或离子中，键角最小的是____\nA. $NH_3$\nB. $NO_3^{-}$\nC. $NF_3$\nD. $NCl_3$\nAnswer:"],
        ["在完全竞争厂商的短期均衡产量上，AR小于SAC但大于AVC，则厂商____\nA. 亏损，立即停产\nB. 亏损，但应继续生产\nC. 亏损，生产\
        或不生产都可以\nD. 获得正常利润，继续生产\nAnswer:"],
        ["α粒子在加速器中被加速，当其质量为静止质量的 3 倍时，其动能为静止能量的____\nA. 2倍\nB. 3倍\nC. 4倍\nD. 5倍\nAnswer:"],
        ["实现一个银行系统，包括存钱、取钱、转账等多项业务，最恰当的资源组合方式是____\nA. 继承\nB. 重载\nC. 组合\nD. 实例化\nAnswer:"],
        ["指令中地址码的长度不仅与主存容量有关，而且还与____有关。\nA. 主存字长\nB. 最小寻址单位\nC. 指令格式\nD. 地址码格式\nAnswer:"],
        ["使用位填充方法，以01111110为位首flag，数据为011011111111111111110010，求问传送时要添加几个0____\nA. 1\nB. 2\nC. 3\n\
        D. 4\nAnswer:"],
        ["设G为平面图，则下面可能不连通的图是____\nA. G的闭合图\nB. G*\nC. (G*)*\nD. (((G)*)*)*\nAnswer:"],
        ["____是高等学校的中心工作，是实现一定的教育目的和人才培养目标的基本途径。\nA. 管理\nB. 教学\nC. 课程\nD. 科研\nAnswer:"],
        ["将三相感应电动机的转子电阻增大一倍，则电机的启动转矩____\nA. 增大\nB. 减小\nC. 不变\nD. 无法判断\nAnswer:"],
        ["工业污水生产周期在8h以内的，每____采样一次。\nA. 2h\nB. 4h\nC. 5h\nD. 6h\nAnswer:"],
        ["根据使用场所不同，闭式细水雾灭火系统可以分为____种形式。\nA. 2\nB. 3\nC. 4\nD. 5\nAnswer:"],
        ["新冠病毒刚刚爆发时检测病患并及时隔离的措施属于____\nA. 注射疫苗\nB. 控制传染源\nC. 切断传播途径\nD. 保护易感人群\nAnswer:"],
        ["下列物质属于合成聚合物的是____\nA. 蛋白质\nB. 人造丝\nC. 人造棉\nD. 聚氯乙烯\nAnswer:"],
        ["下列各句中，没有语病的一句是____\nA. 某些吃惯“大锅饭”的职工对劳动人为制度的革新，切实其实会感到不适应。\nB. “全面建设小康社会”\
        的目标，对于我们感到十分亲热；它已经成为全党天下人民在新世纪中奋斗的行动纲领。\nC. 日本辅弼前去“靖国神社”为东条英机等战争罪犯招魂的\
        反动行径，对于曾饱受侵略战争祸害的中国人民和其他亚洲国家的人民是不克不及容忍的。\nD. 世界重量级拳击冠军易斯接受了女皇颁发的皇家勋章，\
        以表彰他为英国拳击事业做出的贡献。\nAnswer:"],
        ["采用下玻璃温室生产蔬菜、花卉，属于农业分类中的：____\nA. 粗放农业\nB. 自给农业\nC. 种植园农业\nD. 密集农业\nAnswer:"],
        ["中国读书人历来“耻于言商，耻于言利”，而清末状元张謇却放弃仕途，投身于近代工商业。这里反映出的时代观念是____\nA. 实业救国\n\
        B. 工商皆本\nC. 重利轻义\nD. 重商轻农\nAnswer:"],
        ["椭圆$x^{2}+m y^{2}=1$的焦点在x轴上，长轴长是短轴长的两倍，则m的值为____\nA. $\dfrac{1}{4}$\nB. $\dfrac{1}{2}$\n\
        C. 2\nD. 4\nAnswer:"],
        ["提出“电场线”的科学家是____\nA. 库仑\nB. 法拉第\nC. 麦克斯韦\nD. 爱因斯坦\nAnswer:"],
        ["人的本质是____\nA. 永恒不变的\nB. 随主观意志的变化而变化的\nC. 随社会关系的变化而变化的\nD. 随个性的变化而变化\nAnswer:"],
        ["一般来说，一个人对社会的贡献越大，他的____\nA. 社会地位就越高\nB. 个人价值就越高\nC. 社会价值就越高\n\
        D. 自我完善就越高\nAnswer:"],
        ["伪证罪只能发生在____。\nA. 立案侦查以前\nB. 起诉之后\nC. 立案、侦查、起诉和审判过程中\nD. 判决宣告以后\nAnswer:"],
        ["律师应以谁的名义与委托人签订委托代理合同?____\nA. 国家\nB. 律师本人\nC. 律师事务所主任\nD. 律师事务所\nAnswer:"],
        ["关于甲班体育达标测试，三位老师有如下预测：张老师说：“不会所有人都不及格。”李老师说：“有人会不及格。”\
        王老师说：“班长和学习委员都能及格。”如果三位老师中只有一人的预测正确，则以下哪项一定为真?____\nA. 班长和学习委员都没及格。\nB. \
        班长和学习委员都及格了。\nC. 班长及格，但学习委员没及格\nD. 班长没及格，但学习委员及格了。\nAnswer:"],
        ["当今时代的主题是____\nA. 战争与革命\nB. 和平与发展\nC. 开放与合作\nD. 和谐与共赢\nAnswer:"],
        ["经验论是____\nA. 唯心主义的\nB. 唯物主义的\nC. 既可是唯心主义的，也可能是唯物主义的\nD. 二元论的\nAnswer:"],
        ["稳定性是指测量仪器保持其计量特性随时间____的能力。\nA. 响应变化\nB. 慢变化\nC. 稳定\nD. 恒定\nAnswer:"],
        ["正在结西瓜的植株，其吸收的水分主要用于____\nA. 光合作用\nB. 蒸腾作用\nC. 呼吸作用\nD. 吸收作用\nAnswer:"],
        ["下列成语中不涉及化学变化的是____\nA. 星火燎原\nB. 披荆斩棘\nC. 死灰复燃\nD. 百炼成钢\nAnswer:"],
        ["对印度的农业生产有重要意义的是____\nA. 西北季风\nB. 东北季风\nC. 东南季风\nD. 西南季风\nAnswer:"],
        ["形成日后华夏族的主要部落是____\nA. 黄帝、蚩尤部落\nB. 炎帝、蚩尤部落\nC. 黄帝、炎帝、蚩尤部落\nD. 黄帝、炎帝部落\nAnswer:"],
        ["下列图形中有稳定性的是____\nA. 平行四边形\nB. 直角三角形\nC. 长方形\nD. 正方形\nAnswer:"],
        ["能解释“倒影”形成的是____\nA. 光的色散\nB. 光的折射\nC. 光的反射\nD. 光的直线传播\nAnswer:"],
        ["“法定职责必须为，法无授权不可为。”宪法的核心价值追求是____\nA. 依法行政\nB. 保证人民当家作主\nC. 公正司法\n\
        D. 规范权力运行\nAnswer:"],
        ["七届二中全会召开的地点是____\nA. 北平\nB. 延安\nC. 西柏坡\nD. 上海\nAnswer:"],
        ["Unix的软中断机制是____。\nA. 设备中断\nB. 信号量\nC. 系统调用\nD. 信号\nAnswer:"],
        ["正常情况下，小儿颈曲形成的时间是____\nA. 7个月\nB. 6个月\nC. 4个月\nD. 3个月\nAnswer:"],
        ["下列属于生理碱性盐的是____。\nA. $NaNO_3$\nB. $(NH_4)_2SO_4$\nC. $NH_4NO_3$\nD. $MgSO_4$\nAnswer:"],
        ["设随机变量$X$和$Y$相互独立，且X$\sim N(0，1)，Y\sim N(0$，2)，则$D\left(X^2Y^2\right)=$____\nA. 10\n\
        B. 20\nC. 32\nD. 45\nAnswer:"],
        ["下列园林建筑中，____形式优美且不讲究对称布局。\nA. 榭\nB. 轩\nC. 亭\nD. 廊\nAnswer:"],
        ["影响血红蛋白氧饱和度的最主要因素是____\nA. PO2\nB. 血液pH值\nC. PCO2\nD. 血液的温度\nAnswer:"],
        ["下列消费品，属于消费税征税范围的是____。\nA. 合成宝石首饰\nB. 洗发水\nC. 大客车\nD. 轮胎\nAnswer:"],
        ["近代原子核物理学之父是____。\nA. 普朗克\nB. 卢瑟福\nC. 玻尔\nD. 霍金\nAnswer:"],
        ["工商业活动集聚的场所是____，也是从事工商业活动的人群聚居的场所。\nA. 乡村\nB. 郊区\nC. 田园\nD. 城市\nAnswer:"],
        ["稀有核苷酸碱基主要见于____。\nA. DNA\nB. mRNA\nC. tRNA\nD. rRNA\nAnswer:"]
    ]

#gsm8k的校准数据
data_list_3 = [
        "Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for \
        her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck \
        egg. How much in dollars does she make every day at the farmers' market?",
        "A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?",
        "Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  \
        This increased the value of the house by 150%.  How much profit did he make?",
        "James decides to run 3 sprints 3 times a week.  He runs 60 meters each sprint.  How many total meters \
        does he run a week?",
        "Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass \
        costs only 60% of the price. Kylar wants to buy 16 glasses. How much does he need to pay for them?",
        "Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep as Seattle. How many \
        sheep do Toulouse, Charleston, and Seattle have together if Seattle has 20 sheep?",
        "Eliza's rate per hour for the first 40 hours she works each week is $10. She also receives an overtime \
         of 1.2 times her regular hourly rate. If Eliza worked for 45 hours this week, how much are her earnings \
         for this week?",
        "Toula went to the bakery and bought various types of pastries. She bought 3 dozen donuts which cost $68 \
        per dozen, 2 dozen mini cupcakes which cost $80 per dozen, and 6 dozen mini cheesecakes for $55 per dozen. \
        How much was the total cost?",
        "Carlos is planting a lemon tree. The tree will cost $90 to plant. Each year it will grow 7 lemons, which \
        he can sell for $1.5 each. It costs $3 a year to water and feed the tree. How many years will it take \
        before he starts earning money on the lemon tree?",
        "Melanie is a door-to-door saleswoman. She sold a third of her vacuum cleaners at the green house, 2 more to \
        the red house, and half of what was left at the orange house. If Melanie has 5 vacuum cleaners left, how many \
        did she start with?",
        "In a dance class of 20 students, 20% enrolled in contemporary dance, 25% of the remaining enrolled in jazz \
        dance, and the rest enrolled in hip-hop dance. What percentage of the entire students enrolled in \
        hip-hop dance?",
        "Two trains leave San Rafael at the same time. They begin traveling westward, both traveling for 80 miles. \
        The next day, they travel northwards, covering 150 miles. What's the distance covered by each train in \
        the two days?",
        "Jill gets paid $20 per hour to teach and $30 to be a cheerleading coach. If she works 50 weeks a year, \
        35 hours a week as a teacher and 15 hours a week as a coach, what's her annual salary?",
        "Claire makes a 3 egg omelet every morning for breakfast.  How many dozens of eggs will she eat in 4 weeks?",
        "A candle melts by 2 centimeters every hour that it burns. How many centimeters shorter will a candle be \
        after burning from 1:00 PM to 5:00 PM?",
        "Kyle bought last year's best-selling book for $19.50. This is with a 25% discount from the original price. \
        What was the original price of the book?",
        "Darrell and Allen's ages are in the ratio of 7:11. If their total age now is 162, calculate Allen's age 10 \
        years from now.",
        "John takes care of 10 dogs.  Each dog takes .5 hours a day to walk and take care of their business.  How \
        many hours a week does he spend taking care of dogs?",
        "Gretchen has 110 coins. There are 30 more gold coins than silver coins. How many gold coins does \
        Gretchen have?",
        "Terry eats 2 yogurts a day.  They are currently on sale at 4 yogurts for $5.00.  How much does he \
        spend on yogurt over 30 days?",
        "John runs 60 miles a week. He runs 3 days a week.  He runs 3 hours the first day and half as much the \
        other two days he runs.  How fast does he run?"
    ]


def load_tokenizer_and_model(model_path, trust_remote_code=False):
    tokenizer = safe_get_tokenizer_from_pretrained(
        model_path,
        pad_token='<|extra_0|>',
        eos_token='<|endoftext|>',
        padding_side='left',
        trust_remote_code=trust_remote_code
    )
    model = safe_get_model_from_pretrained(
        model_path,
        torch_dtype=torch.float32, trust_remote_code=trust_remote_code
    ).cpu()
    return tokenizer, model


def infer(tokenizer, model, query, model_params=None):
    """
    推理代码
    :param query:
    :param model_params:
    :return:
    """
    inputs = tokenizer(query, return_tensors='pt')
    inputs = inputs.to(model.device)
    with torch.no_grad():
        model_params = model_params if model_params is not None else {}
        pred = model.generate(**inputs, **model_params)
    output = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
    return output


def get_calib_dataset(tokenizer, calib_list):
    calib_dataset = []
    for calib_data in calib_list:
        inputs = tokenizer(calib_data, return_tensors='pt').to("cpu")
        calib_dataset.append([inputs.data['input_ids']])
    return calib_dataset


def check_device_type(value):
    valid_device_types = {"npu", "cpu"}
    if value in valid_device_types:
        return value
    else:
        raise argparse.ArgumentTypeError(f"{value} is not a valid device type. Choose from {valid_device_types}")


def print_args(args):
    for arg in vars(args):
        value = getattr(args, arg)
        value_type = type(value)
        logging.info(f"Argument: {arg}, Value: {value}, Type: {value_type}")


def parse_arguments():
    store_true = 'store_true'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help="model and tokenizer path")
    parser.add_argument('--save_directory', type=str, help="quant weight path")
    parser.add_argument("--w_bit", type=int, default=8)
    parser.add_argument("--a_bit", type=int, default=8)
    parser.add_argument("--act_method", type=int, default=3)
    parser.add_argument('--disable_level', type=str, help="disable_level L0~L5", default='L0')
    parser.add_argument('--device_type', type=check_device_type, required=True, help="device type npu/cpu")
    parser.add_argument("--data_list_index", type=int, default=0)
    parser.add_argument(
        '--disable_names',
        nargs='+',
        default=["lm_head"], required=True)
    parser.add_argument('--anti_method', type=str, default=None)
    parser.add_argument('--w_sym', action=store_true)
    args = parser.parse_args()
    # 查看输出 print_args
    return args

selected_list = {
    1 : data_list_1,
    2 : data_list_2,
    3 : data_list_3
}

if __name__ == '__main__':
    args = parse_arguments()
    input_dict = {
        "model_path":args.model_path,
        "trust_remote_code" : False
    }
    tokenizer, model = load_tokenizer_and_model(**input_dict)
    data_list = selected_list.get(args.data_list_index)
    if data_list is None:
        raise ValueError(f"Selected list is None, invalid key. {args.data_list_index}")
    dataset_calib = get_calib_dataset(tokenizer, data_list)
    disable_names = args.disable_names
    if args.anti_method:
        # dev_type="npu", dev_id=0  如果需要使用npu进行量化
        anti_config = AntiOutlierConfig(
            anti_method=args.anti_method,
            dev_type=args.device_type
        )
        anti_outlier = AntiOutlier(model, calib_data=dataset_calib, cfg=anti_config)
        anti_outlier.process()
    quant_config = QuantConfig(
        w_bit=args.w_bit,  # 权重量化位数
        a_bit=args.a_bit,  # 激活值量化位数
        disable_names=disable_names,  # 不做量化的层（通常是空list）
        dev_type=args.device_type,
        act_method=args.act_method,  # 激活量化方法，建议方法3（1：min-max；2：histogram；3：自动混合量化）
        pr=1.0,  # 量化正则百分比，建议0.5
        w_sym=args.w_sym,  # 对称/非对称量化，True为对称量化，False为非对称量化
        mm_tensor=False  # 权重量化粒度，True为per-tensor量化，False为per-channel量化（大模型场景建议False）
    )
    calibrator = Calibrator(
        model,
        quant_config,
        calib_data=dataset_calib,
        disable_level=args.disable_level  # 自动回退等级，根据精度损失程度增加不量化的层（L0~L5，L0为不回退，精度损失明显时可适当提升等级）
    )

    calibrator.run()  # 执行PTQ量化校准

    calibrator.save(args.save_directory, save_type=["safe_tensor"])   





