// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion

const lightCodeTheme = require('prism-react-renderer').themes.github;
const darkCodeTheme = require('prism-react-renderer').themes.dracula;

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'ParPy',
  tagline: 'A Python library providing configurable parallelization of Python code.',
  url: 'https://parpy-lib.github.io',
  baseUrl: '/',
  projectName: 'ParPy',
  organizationName: 'parpy-lib',
  trailingSlash: false,
  deploymentBranch: 'gh-pages',
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'throw',
  /*favicon: 'img/favicon.ico',*/

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          breadcrumbs: false,
        },
        blog: false,
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      docs : {
        sidebar : {
          hideable: true,
        }
      },
      navbar: {
        title: 'ParPy',
        items: [
          {
            to: 'installation',
            position: 'left',
            label: 'Installation',
          },
          {
            type: 'doc',
            docId: 'root',
            position: 'left',
            label: 'Documentation',
          },
          {
            href: 'https://github.com/parpy-lib/',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Documentation',
            items: [
              {
                label: 'Tutorials',
                to: '/docs/tutorials',
              },
              {
                label: 'Reference',
                to: '/docs/reference',
              },
            ],
          },
          {
            title: 'Links',
            items: [
              {
                label: 'GitHub',
                href: 'https://github.com/parpy-lib/',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Lars Hummelgren`,
      },
      prism: {
      additionalLanguages: ['bash'],
        theme: lightCodeTheme,
        darkTheme: darkCodeTheme,
      },
    }),
};

module.exports = config;
