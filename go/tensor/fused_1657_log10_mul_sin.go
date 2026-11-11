package tensor

import "fmt"

func Fused1657Log10MulSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_mul_sin: nil operand")
	}
	r := x.Mul(y)
	r = r.Log10()
	r = r.Sin()
	return r, nil
}
func Fused1657Log10MulSinName() string { return "log10_mul_sin" }
