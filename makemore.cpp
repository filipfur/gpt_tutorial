#include <cstddef>
#include <cstdint>
#include <string_view>
#include <iostream>
#include <numeric>
#include <algorithm>

#define STB_IMAGE_WRITE_IMPLEMENTATION

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#include "stb_image_write.h"
#pragma clang diagnostic pop

#include "matrix.h"
#include "firstnames.hpp"
//#define CYCLES_ENABLE
//#include "hex/cycles.h"

#include "hex/seed.h"
#include "hex/iterations.h"

#include "tensor.h"

#include <set>
#include <iostream>

#define print_int(var) std::cout << #var << ": " << static_cast<int>(var) << std::endl;
#define print_var(var) std::cout << #var << ": " << var << std::endl;

constexpr firstnames fn;

constexpr char to_lower(char c)
{
    char offset = 'a' - 'A';
    return (c >= 'A' && c <= 'Z') ? (c + offset) : c;
}

static_assert(to_lower('a') == 'a');
static_assert(to_lower('z') == 'z');
static_assert(to_lower('A') == 'a');
static_assert(to_lower('Z') == 'z');

struct Text_Info
{
    constexpr Text_Info(const std::string_view& view_) : view{view_}, num_chars{view.size()}, occurances{}, num_tokens{0}
    {
        for(auto c : view)
        {
            ++occurances[to_lower(c)];
        }
        for(auto val : occurances)
        {
            if(val > 0)
            {
                ++num_tokens;
            }
        }
    }

    std::string_view view;
    size_t num_chars;
    std::array<uint32_t, 256> occurances;
    size_t num_tokens;
};

template <std::size_t N>
struct Tokenizer
{
    constexpr Tokenizer(const Text_Info& text_info) : char_to_index{}, index_to_char{}
    {
        size_t index{0};
        for(size_t i{0}; i < text_info.occurances.size(); ++i)
        {
            if(text_info.occurances[i] > 0)
            {
                index_to_char[index] = to_lower(static_cast<char>(i));
                char_to_index[to_lower(i)] = index;
                ++index;
            }
        }
    }

    uint8_t to_index(char c) const
    {
        return char_to_index[c];
    }

    char to_char(uint8_t i) const
    {
        return index_to_char[i];
    }

    std::array<uint8_t, 256> char_to_index;
    std::array<char, N> index_to_char;
};

template <typename T>
void printGradientDecents(Tensor<T>& root)
{
    std::set<Tensor<T>*> q;
    //q.insert(q.end(), {root._lhs, root._rhs});
    q.insert(root._lhs);
    q.insert(root._rhs);
    std::cout << "gradient decents:" << std::endl;
    while(!q.empty())
    {
        auto ptr = *q.begin();

        std::cout << ptr->_gradient << std::endl;
        if(ptr->_lhs && ptr->_rhs)
        {
            //q.insert(q.end(), {ptr->_lhs, ptr->_rhs});
            q.insert(ptr->_lhs);
            q.insert(ptr->_rhs);
        }
        q.erase(q.begin());
    }
}

int main(int /*argc*/, char* /*argv*/[])
{
    //CYCLES_START(init, 1);
    constexpr Text_Info text_info(fn.text);
    constexpr Tokenizer<text_info.num_tokens> tokenizer{text_info};
    //CYCLES_END(init);
    //CYCLES_ALL(init);

    constexpr auto n = text_info.num_tokens;
    printf("num_chars=%zu num_tokens=%zu\n", text_info.num_chars, text_info.num_tokens);

    print_var(text_info.num_tokens);
    print_var(tokenizer.to_char(6));
    print_int(tokenizer.to_index('d'));

    Matrix<n, n, float> heat;
    heat.fill(1.0f);

    for(size_t i{1}; i < text_info.view.size(); ++i)
    {
        auto a = tokenizer.to_index(to_lower(text_info.view[i - 1]));
        auto b = tokenizer.to_index(to_lower(text_info.view[i]));
        heat[a][b] += 1.0f;
    }

    std::string_view eval_str{R"(anton
filip
astrid
)"};

    for(auto& row : heat._data)
    {
        auto sum = hex::iter_sum(row._data);
        std::transform(std::begin(row), std::end(row), std::begin(row), [sum](auto val) { return val / sum; });
    }

    float log_likelyhood{0.0f};
    int ncount{0};
    for(size_t i{1}; i < eval_str.length(); ++i)
    {
        auto a = tokenizer.to_index(to_lower(eval_str[i - 1]));
        auto b = tokenizer.to_index(to_lower(eval_str[i]));
        float prob = heat[a][b];
        float logprob = std::logf(prob);
        log_likelyhood += logprob;
        ++ncount;
        printf("%c%c: %.4f %.4f\n", eval_str[i - 1], eval_str[i], prob, logprob);
    }
    printf("nll=%.1f\n", -log_likelyhood / ncount);

    Matrix<n, n, uint8_t> imgData;
    for(size_t r{0}; r < n; ++r)
    {
        auto factor = 255.0f / (*std::max_element(std::begin(heat[r]), std::end(heat[r])));//; std::accumulate(std::begin(heat[r]), std::end(heat[r]), 0.0f);
        std::transform(std::begin(heat[r]), std::end(heat[r]), std::begin(imgData[r]), [factor](auto val) -> uint8_t {
            return val * factor;
        });
    }

    int channels = 1;
    stbi_write_png("heat.png",
        n,
        n,
        channels,
        imgData._data.data(),
        n * channels);

    for(size_t i{0}; i < n; ++i)
    {
        printf(" '%c'", tokenizer.to_char(i));
    }
    std::cout << std::endl;

    //static std::array arr = {0.2f, 0.7f, 0.1f};
    static std::default_random_engine generator;
    static std::uniform_real_distribution distribution{0.0, 1.0};
    print_var(heat[0]);
    std::cout << "names: " << std::endl;
    size_t index{0};
    for(int i{0}; i < 8; ++i)
    {
        for(int i{0}; i < 128; ++i)
        {
            index = hex::multinomial(generator, distribution, heat[index]._data);
            std::cout << tokenizer.to_char(index);
            if(index == 0)
            {
                break;
            }
        }
        index = 0;
    }
    std::cout << std::endl;


    // neural net


    Matrix<n, n> W;
    static std::normal_distribution<float> ndist;
    W.fill_rand(generator, ndist);

    Vector<3> v1{{1, 2, 0}};
    Vector<3> v2{{7, 7, 7}};
    print_var(v1 * v2);
    print_var(v1.dot(v2))
    assert(v1.dot(v2) == v2.dot(v1));

    print_var((Vector<3,float>{{1.0f, 0.0f, 0.0f}}.dot(Vector<3,float>{{0.5f,0.0f,0.0f}})));


    Tensor<Matrix<2,2>> t0;
    t0._value.fill(1.0f);
    Tensor<Matrix<2,2>> t1;
    t1._value.fill(2.0f);

    Tensor<Matrix<2,2>> t2{t0 + t1};
    t2._gradient.fill(1.0f);

    print_var(t2._value);
    print_var(t2._gradient);

    t2.backprop();

    print_var(t0._gradient);
    print_var(t1._gradient);

    {
        Matrix m0 = matrix_init<2, 2, float>(
            vector_init<float>(3.0f, 5.0f),
            vector_init<float>(9.0f, 7.0f)
        );
        print_var(m0);

        Vector v0 = vector_init<float>(1.0f, -1.0f, 2.0f, 0.0f, 3.0f, -2.0f);
        print_var(v0);
        print_var(_relu(v0));
        Vector v1 = vector_fill<3, float>(7.0f);
        print_var(v1);
    }



    {
        Tensor a{-4.0f};
        Tensor b{2.0f};
        //Tensor c{a + b};
        Tensor exp{3.0f};
        Tensor d{a * b + b.raise(exp)};

        Tensor bpa = b + a;

        Tensor scalar{2.0f};

        Tensor dd = d + d * scalar + bpa.relu();

        Tensor bma = b - a;

        Tensor ddd = dd + dd * 3.0f + bma.relu();

        //Tensor e{b.raise(exp)};
        //Tensor f{d + e};
        //Tensor g{d + e};

        ddd._gradient = 1.0f;
        ddd.backprop();

        printGradientDecents(d);
    }

    /*Matrix<2, 2, weight<float>> a{{
        Vector<2,weight<float>>{{weight<float>{2.0f}, weight<float>{2.0f}}},
        Vector<2,weight<float>>{{weight<float>{2.0f}, weight<float>{2.0f}}}
    }};
    Matrix<2, 2, weight<float>> b{{
        Vector<2,weight<float>>{{weight<float>{1.0f}, weight<float>{2.0f}}},
        Vector<2,weight<float>>{{weight<float>{3.0f}, weight<float>{4.0f}}}
    }};*/

    //print_var(a[0].dot(b[0]));

    return 0;
}