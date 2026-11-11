package tensor

import "fmt"

func Fused1639Log10MatmulAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_matmul_abs: nil operand")
	}
	r := x.MatMul(y)
	r = r.Log10()
	r = r.Abs()
	return r, nil
}
func Fused1639Log10MatmulAbsName() string { return "log10_matmul_abs" }
