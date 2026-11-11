package tensor

import "fmt"

func Fused1704Log10MatmulLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_matmul_log2: nil operand")
	}
	r := x.MatMul(y)
	r = r.Log10()
	r = r.Log2()
	return r, nil
}
func Fused1704Log10MatmulLog2Name() string { return "log10_matmul_log2" }
