package tensor

import "fmt"

func Fused0089Log10Matmul(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_matmul: nil operand")
	}
	r := x.MatMul(y)
	r = r.Log10()
	return r, nil
}
func Fused0089Log10MatmulName() string { return "log10_matmul" }
