#pragma once

#include <cstdint>
#include <cmath>
#include "matrix.h"

/*template <typename T>
struct weight
{
    constexpr weight() : value{}, a{nullptr}, b{nullptr}, gradient{} {}

    weight(T value_, const weight<T>* a_, const weight<T>* b_) : value{value_}, a{a_}, b{b_}, gradient{} {}

    constexpr weight(T value_) : weight(value_, nullptr, nullptr) {}

    weight<T> operator+(const weight<T>& other) const
    {
        return weight<T>{value + other.value, this, &other};
    }

    weight<T> operator*(const weight<T>& other) const
    {
        return weight<T>{value * other.value, this, &other};
    }

    T value;
    const weight<T>* a;
    const weight<T>* b;
    T gradient;
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const weight<T>& w)
{
    os << w.value;
    return os;
}*/


template <typename T>
T _pow(T a, T b) noexcept
{
    return a * b;
}

template<>
float _pow(float a, float b) noexcept
{
    return powf(a, b);
}

template <typename T>
T _pow_inv(T a, T b) noexcept
{
    return a * b;
}

template<>
int _pow_inv(int a, int b) noexcept
{
    return powf(a, b - 1);
}

template<>
float _pow_inv(float a, float b) noexcept
{
    return powf(a, b - 1.0f);
}

template <typename T>
struct Tensor
{
    enum Function {
        NOP,
        ADD,
        SUB,
        MUL,
        POW_F,
        RELU,
        LOG
    };

    constexpr Tensor() : _value{}, _gradient{}, _lhs{nullptr}, _rhs{nullptr}, _function{NOP} {}
    constexpr Tensor(T value) : _value{value}, _gradient{}, _lhs{nullptr}, _rhs{nullptr}, _function{NOP} {}
    constexpr Tensor(T value, Tensor* lhs, Tensor* rhs, Function function) : _value{value}, _gradient{}, _lhs{lhs}, _rhs{rhs}, _function{function} {}

    Tensor operator+(const Tensor& other)
    {
        return {_value + other._value, const_cast<Tensor*>(this), const_cast<Tensor*>(&other), ADD};
    }

    Tensor operator-(const Tensor& other)
    {
        return {_value - other._value, const_cast<Tensor*>(this), const_cast<Tensor*>(&other), SUB};
    }

    Tensor operator*(const Tensor& other)
    {
        return {_value + other._value, const_cast<Tensor*>(this), const_cast<Tensor*>(&other), MUL};
    }

    Tensor raise(const Tensor& other)
    {
        return {powf(_value, other._value), const_cast<Tensor*>(this), const_cast<Tensor*>(&other), POW_F};
    }

    Tensor relu()
    {
        return {_relu(_value), const_cast<Tensor*>(this), nullptr, RELU};
    }

    Tensor log()
    {
        return {_log(_value), const_cast<Tensor*>(this), nullptr, LOG};
    }

    void backprop()
    {
        if(_lhs != nullptr && _rhs != nullptr)
        {
            switch(_function)
            {
                case ADD:
                    _lhs->_gradient = _lhs->_gradient + _gradient;
                    _rhs->_gradient = _rhs->_gradient + _gradient;
                    break;
                case MUL:
                    _lhs->_gradient = _lhs->_gradient + _rhs->_value * _gradient;
                    _rhs->_gradient = _rhs->_gradient + _lhs->_value * _gradient;
                    break;
                case POW_F:
                    _lhs->_gradient = _lhs->_gradient + _rhs->_value * _pow_inv(_lhs->_value, _rhs->_value) * _gradient;
                    break;
                case RELU:
                    _lhs->_gradient = _lhs->_gradient + is_negative(_value) * _gradient;
                    break;
                case LOG:
                    _lhs->_gradient = _lhs->_gradient + ln_derived(_gradient);
                default:
                    break;
            }
            _lhs->backprop();
            _rhs->backprop();
        }
    }

    T _value;
    T _gradient;
    Tensor* _lhs;
    Tensor* _rhs;
    Function _function;

};