package tensor

import "fmt"

func Fused1644Log10MatmulSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_matmul_sqrt: nil operand")
	}
	r := x.MatMul(y)
	r = r.Log10()
	r = r.Sqrt()
	return r, nil
}
func Fused1644Log10MatmulSqrtName() string { return "log10_matmul_sqrt" }
