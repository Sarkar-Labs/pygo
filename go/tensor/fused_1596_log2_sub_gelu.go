package tensor

import "fmt"

func Fused1596Log2SubGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_sub_gelu: nil operand")
	}
	r := x.Sub(y)
	r = r.Log2()
	r = r.GELU()
	return r, nil
}
func Fused1596Log2SubGeluName() string { return "log2_sub_gelu" }
