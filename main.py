import os
os.environ['KIVY_NO_ENV_CONFIG'] = '1'

import threading
from io import BytesIO

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from kivy.uix.image import Image as KivyImage
from kivy.uix.scrollview import ScrollView
from kivy.metrics import dp

from kivymd.app import MDApp
from kivymd.uix.screen import MDScreen
from kivymd.uix.screenmanager import MDScreenManager
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.gridlayout import MDGridLayout
from kivymd.uix.label import MDLabel
from kivymd.uix.textfield import MDTextField
from kivymd.uix.button import MDRaisedButton, MDFlatButton
from kivymd.uix.card import MDCard
from kivymd.uix.toolbar import MDTopAppBar
from kivymd.uix.spinner import MDSpinner

import analysis as an

ACCENT   = "Blue"
CARD_BG  = [0.141, 0.153, 0.173, 1]
TEXT_PRI = [1, 1, 1, 1]
TEXT_SEC = [0.627, 0.659, 0.753, 1]
BLUE_K   = [0.290, 0.565, 0.886, 1]
GREEN_K  = [0.31,  0.89,  0.64,  1]
AMBER_K  = [0.96,  0.65,  0.14,  1]
CORAL_K  = [0.886, 0.361, 0.361, 1]
NAV_IDLE = [0.18, 0.19, 0.22, 1]
FILTER_BG = [0.10, 0.20, 0.35, 1]

PLT_CARD = "#242740"
BLUE     = "#4A90E2"
GREEN    = "#50E3A4"
AMBER    = "#F5A623"
CORAL    = "#E25C5C"
PLT_COLS = [BLUE, GREEN, AMBER, CORAL, "#B06BE3", "#5CE2D5"]

# ── Global filter state ──────────────────────────────────────────────────────
current_filter = None   # None = show all data


def set_filter(role):
    global current_filter
    current_filter = role


def clear_filter():
    global current_filter
    current_filter = None


# ── Chart helpers ────────────────────────────────────────────────────────────
def fig_to_buf(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight',
                facecolor=PLT_CARD, edgecolor='none', dpi=130)
    buf.seek(0)
    plt.close(fig)
    return buf


def buf_to_image(buf, h=dp(360)):
    cim = CoreImage(buf, ext='png')
    return KivyImage(texture=cim.texture, allow_stretch=True,
                     keep_ratio=True, size_hint=(1, None), height=h)


def bar_h(labels, values, title, color=BLUE, xlabel="Count"):
    n = len(labels)
    fig, ax = plt.subplots(figsize=(7, max(3, n * 0.45)))
    fig.patch.set_facecolor(PLT_CARD)
    ax.set_facecolor(PLT_CARD)
    y    = np.arange(n)
    bars = ax.barh(y, values, color=color, height=0.6, edgecolor='none')
    ax.set_yticks(y)
    ax.set_yticklabels([str(l) for l in labels], color='white', fontsize=9)
    ax.set_xlabel(xlabel, color='#A0A8C0', fontsize=9)
    ax.set_title(title, color='white', fontsize=11, fontweight='bold', pad=10)
    ax.tick_params(colors='#A0A8C0', labelsize=8)
    ax.spines[:].set_visible(False)
    mx = max(values) if values else 1
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + mx * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{int(val):,}', va='center', color='#A0A8C0', fontsize=8)
    ax.set_xlim(0, mx * 1.18)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig_to_buf(fig)


def bar_salary(labels, values, title):
    n = len(labels)
    fig, ax = plt.subplots(figsize=(7, max(3, n * 0.45)))
    fig.patch.set_facecolor(PLT_CARD)
    ax.set_facecolor(PLT_CARD)
    y    = np.arange(n)
    bars = ax.barh(y, values, color=GREEN, height=0.6, edgecolor='none')
    ax.set_yticks(y)
    ax.set_yticklabels([str(l) for l in labels], color='white', fontsize=9)
    ax.set_title(title, color='white', fontsize=11, fontweight='bold', pad=10)
    ax.tick_params(colors='#A0A8C0', labelsize=8)
    ax.spines[:].set_visible(False)
    mx = max(values) if values else 1
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + mx * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'${val/1000:.0f}K', va='center', color='#A0A8C0', fontsize=8)
    ax.set_xlim(0, mx * 1.2)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1000:.0f}K'))
    ax.invert_yaxis()
    fig.tight_layout()
    return fig_to_buf(fig)


def donut(labels, values, title):
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor(PLT_CARD)
    ax.set_facecolor(PLT_CARD)
    colors = PLT_COLS[:len(labels)]
    _, _, autotexts = ax.pie(
        values, labels=None, autopct='%1.1f%%', colors=colors,
        startangle=90, wedgeprops=dict(edgecolor=PLT_CARD, linewidth=2),
        pctdistance=0.75
    )
    for at in autotexts:
        at.set_color('white')
        at.set_fontsize(9)
    ax.add_patch(plt.Circle((0, 0), 0.5, fc=PLT_CARD))
    ax.legend(labels, loc='lower center', ncol=2, frameon=False,
              fontsize=8, labelcolor='white', bbox_to_anchor=(0.5, -0.08))
    ax.set_title(title, color='white', fontsize=11, fontweight='bold', pad=10)
    fig.tight_layout()
    return fig_to_buf(fig)


def histogram(values, title):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor(PLT_CARD)
    ax.set_facecolor(PLT_CARD)
    ax.hist(values, bins=35, color=BLUE, edgecolor=PLT_CARD, linewidth=0.5)
    med = np.median(values)
    mn  = np.mean(values)
    ax.axvline(med, color=CORAL, linestyle='--', linewidth=1.5, label=f'Median ${med/1000:.0f}K')
    ax.axvline(mn,  color=AMBER, linestyle='--', linewidth=1.5, label=f'Mean ${mn/1000:.0f}K')
    ax.set_title(title, color='white', fontsize=11, fontweight='bold', pad=10)
    ax.set_xlabel('Annual Salary (USD)', color='#A0A8C0', fontsize=9)
    ax.set_ylabel('Count', color='#A0A8C0', fontsize=9)
    ax.tick_params(colors='#A0A8C0', labelsize=8)
    ax.spines[:].set_visible(False)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1000:.0f}K'))
    leg = ax.legend(frameon=False, fontsize=8)
    for t in leg.get_texts():
        t.set_color('white')
    fig.tight_layout()
    return fig_to_buf(fig)


def line(labels, values, title):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor(PLT_CARD)
    ax.set_facecolor(PLT_CARD)
    x = np.arange(len(labels))
    ax.plot(x, values, color=BLUE, linewidth=2, marker='o', markersize=4)
    ax.fill_between(x, values, alpha=0.15, color=BLUE)
    step = max(1, len(labels) // 8)
    ax.set_xticks(x[::step])
    ax.set_xticklabels(labels[::step], rotation=40, ha='right', color='#A0A8C0', fontsize=8)
    ax.set_title(title, color='white', fontsize=11, fontweight='bold', pad=10)
    ax.set_ylabel('Postings', color='#A0A8C0', fontsize=9)
    ax.tick_params(colors='#A0A8C0', labelsize=8)
    ax.spines[:].set_visible(False)
    fig.tight_layout()
    return fig_to_buf(fig)



def kpi_card(label, value, color):
    card = MDCard(orientation='vertical', padding=dp(14), spacing=dp(4),
                  radius=[dp(12)], md_bg_color=CARD_BG, elevation=0,
                  size_hint=(1, None), height=dp(90))
    card.add_widget(MDLabel(text=str(value), halign='center',
                            theme_text_color='Custom', text_color=color,
                            font_style='H5', bold=True))
    card.add_widget(MDLabel(text=label, halign='center',
                            theme_text_color='Custom', text_color=TEXT_SEC,
                            font_style='Caption'))
    return card


def chart_wrap(img, h=dp(380)):
    card = MDCard(orientation='vertical', padding=dp(8), radius=[dp(12)],
                  md_bg_color=CARD_BG, elevation=0,
                  size_hint=(1, None), height=h)
    card.add_widget(img)
    return card


def new_scroll():
    layout = MDBoxLayout(orientation='vertical', spacing=dp(14),
                         padding=[dp(16), dp(10), dp(16), dp(20)],
                         size_hint_y=None)
    layout.bind(minimum_height=layout.setter('height'))
    sv = ScrollView(size_hint=(1, 1), do_scroll_x=False)
    sv.add_widget(layout)
    return sv, layout


def filter_banner(role, on_clear):
    """Blue banner shown at the top of each tab when a filter is active."""
    banner = MDCard(
        orientation='horizontal', padding=[dp(14), dp(10), dp(10), dp(10)],
        spacing=dp(8), radius=[dp(10)], md_bg_color=FILTER_BG, elevation=0,
        size_hint=(1, None), height=dp(52)
    )
    banner.add_widget(MDLabel(
        text=f'Filtered:  {role.title()} Job Profile',
        theme_text_color='Custom', text_color=BLUE_K,
        font_style='Subtitle2', bold=True
    ))
    clear_btn = MDFlatButton(
        text='Clear Filter',
        theme_text_color='Custom',
        text_color=CORAL_K,
        size_hint=(None, None),
        height=dp(36)
    )
    clear_btn.bind(on_release=lambda x: on_clear())
    banner.add_widget(clear_btn)
    return banner



class OverviewScreen(MDScreen):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.name = 'overview'

    def on_enter(self):
        self.refresh()

    def refresh(self):
        self.clear_widgets()
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        role   = current_filter
        stats  = an.get_overview_stats(role)
        rl, rv = an.get_top_roles(role=role)
        reml, remv = an.get_remote_vs_onsite(role)
        tl, tv = an.get_posting_trend(role)

        title_str = f'Job Postings by Role' if not role else f'Role Mix — {role.title()}'
        b_roles = bar_h(rl,   rv,   title_str)
        b_donut = donut(reml, remv, 'Remote vs On-site')
        b_trend = line(tl,    tv,   'Posting Trend Over Time')

        def finish(dt):
            sv, lay = new_scroll()
            avg = stats['avg_salary']
            med = stats['median_salary']

            if role:
                lay.add_widget(filter_banner(role, self._on_clear))

            r1 = MDGridLayout(cols=3, spacing=dp(10), size_hint=(1, None), height=dp(100))
            r1.add_widget(kpi_card('Total Jobs',    f"{stats['total_jobs']:,}",             BLUE_K))
            r1.add_widget(kpi_card('Avg Salary',    f"${avg/1000:.0f}K" if avg else 'N/A', GREEN_K))
            r1.add_widget(kpi_card('Remote %',      f"{stats['remote_pct']:.1f}%",          AMBER_K))

            r2 = MDGridLayout(cols=3, spacing=dp(10), size_hint=(1, None), height=dp(100))
            r2.add_widget(kpi_card('Median Salary', f"${med/1000:.0f}K" if med else 'N/A', CORAL_K))
            r2.add_widget(kpi_card('w/ Salary',     f"{stats['jobs_w_salary']:,}",          [0.69, 0.42, 0.89, 1]))
            r2.add_widget(kpi_card('Top Role',      stats['top_role'],                      [0.36, 0.886, 0.835, 1]))

            lay.add_widget(r1)
            lay.add_widget(r2)
            lay.add_widget(chart_wrap(buf_to_image(b_roles, dp(360)), dp(380)))
            lay.add_widget(chart_wrap(buf_to_image(b_donut, dp(310)), dp(330)))
            lay.add_widget(chart_wrap(buf_to_image(b_trend, dp(300)), dp(320)))
            self.add_widget(sv)

        Clock.schedule_once(finish)

    def _on_clear(self):
        clear_filter()
        MDApp.get_running_app().refresh_all_tabs()


class SkillsScreen(MDScreen):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.name = 'skills'

    def on_enter(self):
        self.refresh()

    def refresh(self):
        self.clear_widgets()
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        role = current_filter
        sl, sv   = an.get_top_skills(15, role)
        sal_l, sal_v = an.get_skills_salary(role)

        t1 = f'Top 15 In-Demand Skills' if not role else f'Top Skills — {role.title()}'
        t2 = f'Skills That Pay the Most' if not role else f'Highest Paying Skills — {role.title()}'
        b_demand = bar_h(sl,    sv,    t1, xlabel='Postings')
        b_salary = bar_salary(sal_l, sal_v, t2) if sal_l else None

        def finish(dt):
            sv2, lay = new_scroll()
            if role:
                lay.add_widget(filter_banner(role, self._on_clear))
            lay.add_widget(chart_wrap(buf_to_image(b_demand, dp(500)), dp(520)))
            if b_salary:
                lay.add_widget(chart_wrap(buf_to_image(b_salary, dp(420)), dp(440)))
            self.add_widget(sv2)

        Clock.schedule_once(finish)

    def _on_clear(self):
        clear_filter()
        MDApp.get_running_app().refresh_all_tabs()


class LocationScreen(MDScreen):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.name = 'location'

    def on_enter(self):
        self.refresh()

    def refresh(self):
        self.clear_widgets()
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        role = current_filter
        cl, cv = an.get_top_cities(10, role)
        rl, rv = an.get_salary_by_role(role)

        t1 = 'Top 10 Hiring Cities'         if not role else f'Top Cities — {role.title()}'
        t2 = 'Average Salary by Role'        if not role else f'Salary by Role — {role.title()}'
        b_cit = bar_h(cl, cv, t1, color=AMBER)
        b_rol = bar_salary(rl, rv, t2) if rl else None

        def finish(dt):
            sv, lay = new_scroll()
            if role:
                lay.add_widget(filter_banner(role, self._on_clear))
            lay.add_widget(chart_wrap(buf_to_image(b_cit, dp(380)), dp(400)))
            if b_rol:
                lay.add_widget(chart_wrap(buf_to_image(b_rol, dp(360)), dp(380)))
            self.add_widget(sv)

        Clock.schedule_once(finish)

    def _on_clear(self):
        clear_filter()
        MDApp.get_running_app().refresh_all_tabs()


class SalaryScreen(MDScreen):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.name = 'salary'

    def on_enter(self):
        self.refresh()

    def refresh(self):
        self.clear_widgets()
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        role = current_filter
        sal_vals    = an.get_salary_distribution(role)
        reml, remv  = an.get_remote_vs_onsite(role)

        t1 = 'Salary Distribution'   if not role else f'Salary Distribution — {role.title()}'
        t2 = 'Remote vs On-site Split'
        b_hist  = histogram(sal_vals, t1) if sal_vals else None
        b_donut = donut(reml, remv, t2)

        def finish(dt):
            sv, lay = new_scroll()
            if role:
                lay.add_widget(filter_banner(role, self._on_clear))
            if b_hist:
                lay.add_widget(chart_wrap(buf_to_image(b_hist,  dp(300)), dp(320)))
            lay.add_widget(chart_wrap(buf_to_image(b_donut, dp(320)), dp(340)))
            self.add_widget(sv)

        Clock.schedule_once(finish)

    def _on_clear(self):
        clear_filter()
        MDApp.get_running_app().refresh_all_tabs()


class SearchScreen(MDScreen):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.name = 'search'
        self._build_ui()

    def _build_ui(self):
        root = MDBoxLayout(orientation='vertical', spacing=dp(12),
                           padding=[dp(16), dp(14), dp(16), dp(20)])
        row = MDBoxLayout(orientation='horizontal', spacing=dp(10),
                          size_hint=(1, None), height=dp(52))
        self.field = MDTextField(
            hint_text='Enter job title (e.g. Data Analyst)',
            mode='rectangle',
            radius=[dp(10), dp(10), dp(10), dp(10)],
            size_hint_x=0.78,
            on_text_validate=self._on_search
        )
        btn = MDRaisedButton(text='Search', size_hint=(0.22, None),
                             height=dp(48), md_bg_color=BLUE_K)
        btn.bind(on_release=self._on_search)
        row.add_widget(self.field)
        row.add_widget(btn)

        self.sv = ScrollView(size_hint=(1, 1), do_scroll_x=False)
        self.rbox = MDBoxLayout(orientation='vertical', spacing=dp(12),
                                size_hint_y=None, padding=[0, dp(8), 0, dp(20)])
        self.rbox.bind(minimum_height=self.rbox.setter('height'))
        self.rbox.add_widget(MDLabel(
            text='Search a job role to see insights.\nAll other tabs will also update to show that role.',
            halign='center', theme_text_color='Custom',
            text_color=TEXT_SEC, font_style='Body1'
        ))
        self.sv.add_widget(self.rbox)
        root.add_widget(row)
        root.add_widget(self.sv)
        self.add_widget(root)

    def _on_search(self, *args):
        role = self.field.text.strip()
        if not role:
            return
        self.rbox.clear_widgets()
        self.rbox.add_widget(MDSpinner(size_hint=(None, None),
                                       size=(dp(46), dp(46)),
                                       pos_hint={'center_x': .5}))
        threading.Thread(target=self._do_search, args=(role,), daemon=True).start()

    def _do_search(self, role):
        data = an.analyze_job_market(role)

        def finish(dt):
            self.rbox.clear_widgets()
            if data is None:
                self.rbox.add_widget(MDLabel(
                    text=f'No results for "{role}".\nTry "Data Analyst" or "Business Analyst".',
                    halign='center', theme_text_color='Custom', text_color=CORAL_K))
                return

            
            set_filter(role)
            MDApp.get_running_app().refresh_other_tabs('search')

            
            header_card = MDCard(
                orientation='vertical', padding=dp(14), spacing=dp(4),
                radius=[dp(10)], md_bg_color=CARD_BG, elevation=0,
                size_hint=(1, None), height=dp(62)
            )
            header_card.add_widget(MDLabel(
                text=f'Showing results for:  {role.title()} Job Profile',
                halign='left',
                theme_text_color='Custom',
                text_color=BLUE_K,
                font_style='Subtitle1',
                bold=True,
                size_hint_y=None,
                height=dp(30)
            ))
            header_card.add_widget(MDLabel(
                text=f'{data["total"]:,} postings found  •  All tabs updated',
                halign='left',
                theme_text_color='Custom',
                text_color=TEXT_SEC,
                font_style='Caption',
                size_hint_y=None,
                height=dp(20)
            ))
            self.rbox.add_widget(header_card)
            

            avg = data['avg_salary']
            r = MDGridLayout(cols=3, spacing=dp(10), size_hint=(1, None), height=dp(100))
            r.add_widget(kpi_card('Postings',   f"{data['total']:,}",             BLUE_K))
            r.add_widget(kpi_card('Avg Salary', f"${avg/1000:.0f}K" if avg > 0 else 'N/A', GREEN_K))
            r.add_widget(kpi_card('Remote %',   f"{data['remote_pct']:.0f}%",     AMBER_K))
            self.rbox.add_widget(r)

            for title, items, color in [
                ('Top Skills',    data['top_skills'],    BLUE_K),
                ('Top Cities',    data['top_cities'],    AMBER_K),
                ('Top Companies', data['top_companies'], GREEN_K),
            ]:
                card = MDCard(orientation='vertical', padding=dp(14), spacing=dp(6),
                              radius=[dp(10)], md_bg_color=CARD_BG, elevation=0,
                              size_hint=(1, None),
                              height=dp(38 + 24 * (len(items) + 1)))
                card.add_widget(MDLabel(text=title, bold=True,
                                        theme_text_color='Custom', text_color=color,
                                        font_style='Subtitle1',
                                        size_hint_y=None, height=dp(30)))
                for item in items:
                    card.add_widget(MDLabel(text=f'  •  {item}',
                                            theme_text_color='Custom', text_color=TEXT_PRI,
                                            font_style='Body2',
                                            size_hint_y=None, height=dp(24)))
                self.rbox.add_widget(card)

            
            clr = MDFlatButton(
                text='Clear Filter — Show All Jobs',
                theme_text_color='Custom',
                text_color=CORAL_K,
                size_hint=(1, None), height=dp(42)
            )
            clr.bind(on_release=lambda x: self._clear())
            self.rbox.add_widget(clr)

        Clock.schedule_once(finish)

    def _clear(self):
        clear_filter()
        self.field.text = ''
        self.rbox.clear_widgets()
        self.rbox.add_widget(MDLabel(
            text='Filter cleared. All tabs now show general data.',
            halign='center', theme_text_color='Custom',
            text_color=GREEN_K, font_style='Body1'
        ))
        MDApp.get_running_app().refresh_other_tabs('search')



class JobApp(MDApp):

    def build(self):
        self.theme_cls.theme_style     = 'Dark'
        self.theme_cls.primary_palette = ACCENT
        self.title = 'SkillScope'

        root = MDBoxLayout(orientation='vertical')

        toolbar = MDTopAppBar(title='SkillScope',
                              orientation='horizontal',
                              size_hint=(1, None),
                              height=dp(64),
                              md_bg_color=CARD_BG,
                              padding=[dp(20), dp(10), dp(20), dp(10)]
                        )
        toolbar.add_widget(MDLabel(
            text="SkillScope",
            halign='center',
            theme_text_color='Custom',
            text_color=BLUE_K,
            font_style="H4",
            bold=True
            ))

        nav = MDBoxLayout(orientation='horizontal', size_hint=(1, None),
                          height=dp(50), md_bg_color=CARD_BG,
                          padding=[dp(4), dp(4), dp(4), dp(4)], spacing=dp(4))

        self.sm = MDScreenManager()
        self._screens = {
            'overview': OverviewScreen(),
            'skills':   SkillsScreen(),
            'location': LocationScreen(),
            'salary':   SalaryScreen(),
            'search':   SearchScreen(),
        }
        for s in self._screens.values():
            self.sm.add_widget(s)

        self.nav_btns = {}
        for label, sname in [('Overview', 'overview'), ('Skills', 'skills'),
                              ('Location', 'location'), ('Salary', 'salary'),
                              ('Search',   'search')]:
            b = MDRaisedButton(text=label, size_hint=(1, None),
                               height=dp(42), elevation=0, font_size='11sp')
            b.bind(on_release=lambda x, s=sname: self._go(s))
            nav.add_widget(b)
            self.nav_btns[sname] = b

        root.add_widget(toolbar)
        root.add_widget(nav)
        root.add_widget(self.sm)
        self._go('overview')
        return root

    def _go(self, name):
        self.sm.current = name
        for sn, btn in self.nav_btns.items():
            btn.md_bg_color = BLUE_K if sn == name else NAV_IDLE

    def refresh_all_tabs(self):
        """Refresh every tab (used when filter is cleared)."""
        for screen in self._screens.values():
            if hasattr(screen, 'refresh'):
                screen.refresh()

    def refresh_other_tabs(self, skip_name):
        """Refresh all tabs except the one specified (used after search)."""
        for name, screen in self._screens.items():
            if name != skip_name and hasattr(screen, 'refresh'):
                screen.refresh()


if __name__ == '__main__':
    JobApp().run()
