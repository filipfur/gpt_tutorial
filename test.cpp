#include <iostream>
#include <array>

#include "hex/iterations.h"

#include "tinyshakespeare.hpp"

#include "matrix.h"

constexpr tinyshakespeare text;

constexpr std::string_view view{text.text};
constexpr auto view_len = view.length();
constexpr auto eval_len = static_cast<size_t>(view_len * 0.1f);
constexpr auto train_len = view_len - eval_len;


constexpr int block_size{8};
constexpr int batch_size{4};

#define print_var(var) std::cout << #var << ": " << var << std::endl;

template <typename Iter, typename T>
void copy_out(Iter first, Iter last)
{
    std::copy(first, last, std::ostream_iterator<T>{std::cout, " "});
}

struct TextInfo
{
    TextInfo()
    {
        for(int i{0}; i < 1500000; ++i)
        {
            if(text.text[i] == '\0')
            {
                no_tokens = i;
                break;
            }
            occurances[text.text[i]] += 1;
        }

        for(int i{0}; i < 256; ++i)
        {
            if(occurances[i] > 0)
            {
                token_number[i] = chars_length;
                chars[chars_length++] = static_cast<char>(i);
            }
        }
    }

    std::array<uint32_t, 256> occurances{};
    std::array<uint8_t, 256> chars{};
    size_t chars_length{0}; 
    std::array<uint8_t, 256> token_number{};
    size_t no_tokens{0};
};

template <std::size_t N>
void create_batch(const std::array<uint8_t, N>& data, Matrix<batch_size, block_size>& xb, Matrix<batch_size, block_size>& yb)
{
    std::array<int, batch_size> ix;
    std::transform(std::begin(ix), std::end(ix), std::begin(ix), [](int) { return rand() % (N - block_size); });

    for(int b{0}; b < batch_size; ++b)
    {
        for(int i{0}; i < block_size; ++i)
        {
            xb[b][i] = data[ix[b] + i];
        }
    }

    for(int b{0}; b < batch_size; ++b)
    {
        for(int i{0}; i < block_size; ++i)
        {
            yb[b][i] = data[ix[b] + i + 1];
        }
    }
}

int main(int /*argc*/, char* /*argv*/[])
{
    srand(1337);

    TextInfo textInfo;

    print_var(view_len);
    print_var(train_len);
    print_var(eval_len);

    std::array<uint8_t, train_len> train_data{};
    std::array<uint8_t, eval_len> eval_data{};
    auto a = std::begin(view);
    auto b = std::next(a, train_len);
    auto c = std::end(view);
    std::transform(a, b, train_data.begin(), [&](char c) { return textInfo.token_number[c]; });
    std::transform(b, c, eval_data.begin(), [&](char c) { return textInfo.token_number[c]; });

    for(int t{0}; t < block_size; ++t)
    {
        std::vector<uint8_t> context{std::begin(train_data), std::next(std::begin(train_data), t + 1)};
        uint8_t target = train_data[t + 1];
        printf("when input is ");
        copy_out<std::vector<uint8_t>::iterator,int>(std::begin(context), std::end(context));
        printf("the target is %d\n", target);
    }

    Matrix<batch_size, block_size> xb;
    Matrix<batch_size, block_size> yb;

    create_batch(train_data, xb, yb);

    print_var(xb);
    print_var(yb);

    // TBC 24 min https://www.youtube.com/watch?v=kCc8FmEb1nY

    return 0;
}