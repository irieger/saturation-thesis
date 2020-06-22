survey_data_set = {
    color_spaces: [ { text: 'sRGB - pure OETF', mode: 'images', path: 'srgb_oetf' },
                    { text: 'sRGB - tonemapped', mode: 'images', path: 'srgb_tonemapped' },
                    { text: 'Display P3 - pure OETF', mode: 'images', path: 'disp3_oetf' },
                    { text: 'Display P3 - tonemapped', mode: 'images', path: 'disp3_tonemapped' },
                    { text: 'Graphs', mode: 'graphs', path: 'graphs' }
                ],
    image_sets: {
        default:          [ { text: 'Color Checker (Gretag 24 patches)', path: 'gretag_cc' } ],
        graphs:           [ { text: 'Primaries', path: 'primaries' } ]
    },

    saturation: [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
    
    color_models: [
        { text: 'YCbCr ITU-R BT.709', path: 'bt709' },
        { text: 'YCbCr ITU-R BT.2020', path: 'bt2020' },
        { text: 'YCbCr ITU-R BT.2020 (const luma / ICtCp)', path: 'bt2020ictcp' }
    ]
}