<template>
    <form @submit.prevent="onSubmit">
        <label for="new-todo-input"> What needs to be done? </label>
        <input type="text" id="new-todo-input" name="new-todo" autocomplete="off" v-model.lazy.trim="label" />
        <button type="submit">Add</button>
    </form>
</template>
  
<script>
export default {
    data() {
        return {
            label: ""
        }
    },
    methods: {
        onSubmit() {
            if(this.label === ""){
                return
            }
            console.log("form submitted label = " + this.label)
            this.$emit("todo-added", this.label)
            this.label = ""
        }
    }
};
</script>
