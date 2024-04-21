#pragma once

#include <cstddef>
#include <array>
#include <iostream>
#include <random>
#include <algorithm>

template <typename IterIn, typename IterOut>
void iter_addition(IterIn iterIn1, IterIn iterIn1End, IterIn iterIn2, IterOut iterOut)
{
    while(iterIn1 != iterIn1End)
    {
        *iterOut = (*iterIn1) + (*iterIn2);
        ++iterIn1;
        ++iterIn2;
        ++iterOut;
    }
}

template <typename IterIn, typename IterOut>
void iter_multiplication(IterIn iterIn1, IterIn iterIn1End, IterIn iterIn2, IterOut iterOut)
{
    while(iterIn1 != iterIn1End)
    {
        *iterOut = (*iterIn1) * (*iterIn2);
        ++iterIn1;
        ++iterIn2;
        ++iterOut;
    }
}

template <typename IterIn, typename T, typename IterOut>
void iter_scale(IterIn iterIn1, IterIn iterIn1End, T val, IterOut iterOut)
{
    while(iterIn1 != iterIn1End)
    {
        *iterOut = (*iterIn1) * val;
        ++iterIn1;
        ++iterOut;
    }
}

template <typename Iter>
void iter_increment(Iter first, Iter last)
{
    for(; first != last; ++first)
    {
        ++(*first);
    }
}

template <typename Iter>
void iter_decrement(Iter first, Iter last)
{
    for(; first != last; ++first)
    {
        --(*first);
    }
}

template <typename IterIn, typename IterOut>
void iter_relu(IterIn first, IterIn last, IterOut out);

template <typename IterIn, typename IterOut>
void iter_is_negative(IterIn first, IterIn last, IterOut out);

template <std::size_t N, typename T=float>
struct Vector
{
    Vector() : _data{} {}

    Vector(const std::array<T, N>& data) : _data{data} {}
    Vector(std::array<T, N>&& data) : _data{std::move(data)} {}

    ~Vector()
    {
    }
    
    Vector(const Vector& other)
    {
        std::copy(other.begin(), other.end(), begin());
    }

    Vector& operator=(const Vector& other)
    {
        std::copy(other.begin(), other.end(), begin());
    }
    
    Vector(Vector&& other)
    {
        _data = std::move(other._data);
    }

    Vector& operator=(Vector&& other)
    {
        _data = std::move(other._data);
        return *this;
    }

    std::array<T, N>::iterator begin()
    {
        return _data.begin();
    }

    std::array<T, N>::iterator end()
    {
        return _data.end();
    }

    std::array<T, N>::const_iterator begin() const
    {
        return _data.begin();
    }

    std::array<T, N>::const_iterator end() const
    {
        return _data.end();
    }

    void fill(T t)
    {
        std::fill(std::begin(_data), std::end(_data), t);
    }

    template <typename G>
    void fill_rand(std::default_random_engine& generator, G& distribution)
    {
        std::transform(std::begin(_data), std::end(_data), std::begin(_data), [&generator, &distribution](auto){
            return distribution(generator);
        });
    }

    Vector<N,T> operator+(const Vector<N,T>& other) const
    {
        Vector<N,T> result;
        iter_addition(_data.begin(), _data.end(), other._data.begin(), result._data.begin());
        return result;
    }

    Vector<N,T> operator*(const Vector<N,T>& other) const
    {
        Vector<N,T> result;
        iter_multiplication(_data.begin(), _data.end(), other._data.begin(), result._data.begin());
        return result;
    }

    Vector<N,T> operator*(T val) const
    {
        Vector<N,T> result;
        iter_scale(_data.begin(), _data.end(), val, result._data.begin());
        return result;
    }

    Vector<N,T>& operator++()
    {
        iter_increment(_data.begin(), _data.end());
        return *this;
    }

    Vector<N,T>& operator--()
    {
        iter_decrement(_data.begin(), _data.end());
        return *this;
    }

    T dot(const Vector<N,T>& other) const
    {
        static constexpr T init{};
        return std::inner_product(std::begin(_data), std::end(_data), std::begin(other._data), init);
    }

    /*Vector<N,T> relu() const
    {
        Vector<N,T> result;
        iter_relu(std::begin(_data), std::end(_data), std::begin(result));
        return result;
    }*/

    T& operator[](std::size_t index)
    {
        return _data[index];
    }

    const T& operator[](std::size_t index) const
    {
        return _data[index];
    }

    std::array<T, N> _data;
};

template <std::size_t M, std::size_t N, typename T=float>
struct Matrix
{
    Matrix() : _data{}
    {

    }

    Matrix(const std::array<Vector<N, T>, M>& data) : _data{data} {}
    Matrix(std::array<Vector<N, T>, M>&& data) : _data{std::move(data)} {}

    void fill(T t)
    {
        for(auto& v : _data)
        {
            v.fill(t);
        }
    }

    template <typename G>
    void fill_rand(std::default_random_engine& generator, G& distribution)
    {
        for(auto& v : _data)
        {
            v.fill_rand(generator, distribution);
        }
    }

    Vector<N, T>& operator[](std::size_t index)
    {
        return _data[index];
    }

    const Vector<N, T>& operator[](std::size_t index) const
    {
        return _data[index];
    }

    Matrix<M,N,T> operator+(const Matrix<M,N,T>& other) const
    {
        Matrix<M,N,T> result;
        iter_addition(_data.begin(), _data.end(), other._data.begin(), result._data.begin());
        return result;
    }

    Matrix<M,N,T> operator*(const Matrix<M,N,T>& other) const
    {
        Matrix<M,N,T> result;
        iter_multiplication(_data.begin(), _data.end(), other._data.begin(), result._data.begin());
        return result;
    }

    Matrix<M,N,T> operator*(T val) const
    {
        Matrix<M,N,T> result;
        iter_scale(_data.begin(), _data.end(), val, result._data.begin());
        return result;
    }

    /*Matrix<M,N,T> relu() const
    {
        Matrix<M,N,T> result;
        iter_relu(std::begin(_data), std::end(_data), std::begin(result));
        return result;
    }*/

    Matrix<M,N,T>& operator++()
    {
        iter_increment(_data.begin(), _data.end());
        return *this;
    }

    Matrix<M,N,T>& operator--()
    {
        iter_decrement(_data.begin(), _data.end());
        return *this;
    }

    std::array<Vector<N, T>, M> _data;
};

template <std::size_t N, typename T>
std::ostream& operator<<(std::ostream& os, const Vector<N, T>& v)
{
    std::string delim = "";
    for(auto t : v._data)
    {
        std::cout << delim << t;
        delim = " ";
    }
    std::cout << std::endl;
    return os;
}

template <std::size_t M, std::size_t N, typename T>
std::ostream& operator<<(std::ostream& os, const Matrix<M, N, T>& m)
{
    std::cout << std::endl;
    for(const auto& v : m._data)
    {
        std::cout << v;
    }
    return os;
}

template <typename T>
T _relu(const T& val)
{
    T result;
    iter_relu(std::begin(val._data), std::end(val._data), std::begin(result._data));
    return result;
}

template <>
float _relu(const float& val)
{
    return val < 0.0f ? 0.0f : val;
}

template <>
int _relu(const int& val)
{
    return val < 0 ? 0 : val;
}

template <typename T>
T is_negative(const T& val)
{
    T result;
    iter_is_negative(std::begin(val._data), std::end(val._data), std::begin(result._data));
    return result;
}

template<>
float is_negative(const float& val)
{
    return val < 0.0f ? 1.0f : 0.0f;
}

template<>
int is_negative(const int& val)
{
    return val < 0 ? 1 : 0;
}

template <std::size_t N, typename T>
constexpr Vector<N, T> vector_fill(T val)
{
    std::array<T, N> arr{};
    std::fill(std::begin(arr), std::end(arr), val);
    return {arr};
}

template <typename T, typename ... Args>
Vector<sizeof...(Args), T> vector_init(Args ... arg)
{
    return {{arg...}};
}

template <std::size_t M, std::size_t N, typename T, typename ... Args>
Matrix<M, N, T> matrix_init(Args ... arg)
{
    return {{arg...}};
}

template <typename IterIn, typename IterOut>
void iter_relu(IterIn first, IterIn last, IterOut out)
{
    for(; first != last; ++first)
    {
        *out = _relu(*first);
        ++out;
    }
}

template <typename IterIn, typename IterOut>
void iter_is_negative(IterIn first, IterIn last, IterOut out)
{
    for(; first != last; ++first)
    {
        *out = is_negative(*first);
        ++out;
    }
}